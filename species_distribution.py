import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import folium
from folium import plugins
import json
import pickle
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class SpeciesDistributionModeler:
    def __init__(self, db_path="biodiversity_data/indonesia_biodiversity.db", 
                 results_dir="sdm_results"):
        self.db_path = db_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Indonesia bounds for prediction grid
        self.indonesia_bounds = {
            'min_lat': -11.0,
            'max_lat': 6.0,
            'min_lng': 95.0,
            'max_lng': 141.0
        }
        
        # Environmental variables for modeling (updated to match what we actually have)
        self.env_variables = [
            'bio1', 'bio2', 'bio3', 'bio4', 'bio5', 'bio6', 'bio7', 'bio8',
            'bio9', 'bio10', 'bio11', 'bio12', 'bio13', 'bio14', 'bio15',
            'bio16', 'bio17', 'bio18', 'bio19', 'elevation', 'slope',
            'aspect', 'distance_to_coast', 'population_density'
        ]
        
        self.species_models = {}
        self.environmental_data = None
        self.prediction_grid = None
        
    def check_environmental_data(self):
        """Check what environmental data is actually available"""
        print("🔍 Checking available environmental data...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='environmental_data'")
        if not cursor.fetchone():
            print("❌ No environmental_data table found!")
            print("   Please run the environmental data processor first.")
            return False
        
        # Check columns
        cursor.execute("PRAGMA table_info(environmental_data)")
        available_columns = [row[1] for row in cursor.fetchall()]
        
        # Check row count
        cursor.execute("SELECT COUNT(*) FROM environmental_data")
        row_count = cursor.fetchone()[0]
        
        print(f"✅ Environmental data table found with {row_count:,} records")
        print(f"📋 Available columns: {available_columns}")
        
        # Update env_variables to only include available columns
        self.env_variables = [col for col in self.env_variables if col in available_columns]
        print(f"✅ Will use {len(self.env_variables)} environmental variables for modeling")
        
        conn.close()
        return row_count > 0
        
    def load_environmental_data(self):
        """Load environmental data from database"""
        print("📊 Loading environmental data...")
        
        if not self.check_environmental_data():
            return None
        
        conn = sqlite3.connect(self.db_path)
        
        # Build query with only available columns
        env_vars_str = ', '.join(self.env_variables)
        query = f"""
        SELECT latitude, longitude, {env_vars_str}
        FROM environmental_data
        WHERE latitude IS NOT NULL 
        AND longitude IS NOT NULL
        AND ({' IS NOT NULL OR '.join(self.env_variables[:5])} IS NOT NULL)
        """
        
        self.environmental_data = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"✅ Loaded {len(self.environmental_data)} environmental data points")
        print(f"📊 Environmental variables: {', '.join(self.env_variables)}")
        return self.environmental_data
    
    def get_species_list(self, min_occurrences=20):
        """Get list of species with sufficient occurrence records"""
        print(f"🔍 Finding species with at least {min_occurrences} occurrences...")
        
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
        SELECT species, COUNT(*) as occurrence_count
        FROM occurrences 
        WHERE species IS NOT NULL 
        AND decimalLatitude IS NOT NULL 
        AND decimalLongitude IS NOT NULL
        GROUP BY species
        HAVING COUNT(*) >= {min_occurrences}
        ORDER BY COUNT(*) DESC
        """
        
        species_counts = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"✅ Found {len(species_counts)} species with ≥{min_occurrences} occurrences")
        print(f"Top 10 most common species:")
        for i, (_, row) in enumerate(species_counts.head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['species']:<40} ({row['occurrence_count']:,} records)")
        
        return species_counts['species'].tolist()
    
    def prepare_species_data(self, species_name):
        """Prepare training data for a specific species"""
        print(f"📋 Preparing data for {species_name}...")
        
        if self.environmental_data is None:
            self.load_environmental_data()
        
        if self.environmental_data is None:
            print("❌ No environmental data available")
            return None
        
        conn = sqlite3.connect(self.db_path)
        
        # Get presence points for the species
        presence_query = f"""
        SELECT DISTINCT decimalLatitude as latitude, decimalLongitude as longitude
        FROM occurrences 
        WHERE species = ? 
        AND decimalLatitude IS NOT NULL 
        AND decimalLongitude IS NOT NULL
        AND decimalLatitude BETWEEN {self.indonesia_bounds['min_lat']} AND {self.indonesia_bounds['max_lat']}
        AND decimalLongitude BETWEEN {self.indonesia_bounds['min_lng']} AND {self.indonesia_bounds['max_lng']}
        """
        
        presence_points = pd.read_sql_query(presence_query, conn, params=(species_name,))
        conn.close()
        
        print(f"  Found {len(presence_points)} presence points")
        
        # Generate pseudo-absences
        n_pseudoabsences = min(len(presence_points) * 3, 1000)  # 3:1 ratio, max 1000
        pseudo_absences = self.generate_pseudo_absences(presence_points, n_pseudoabsences)
        
        print(f"  Generated {len(pseudo_absences)} pseudo-absence points")
        
        # Combine presence and absence points
        presence_points['presence'] = 1
        pseudo_absences['presence'] = 0
        
        all_points = pd.concat([presence_points, pseudo_absences], ignore_index=True)
        
        # Match with environmental data
        training_data = self.match_with_environmental_data(all_points)
        
        if training_data is None or len(training_data) < 20:
            print(f"  ⚠️  Insufficient training data ({len(training_data) if training_data is not None else 0} points)")
            return None
        
        print(f"  ✅ Prepared {len(training_data)} training points")
        return training_data
    
    def generate_pseudo_absences(self, presence_points, n_pseudoabsences):
        """Generate pseudo-absence points using random sampling"""
        np.random.seed(42)  # For reproducibility
        
        # Create buffer around presence points to avoid pseudo-absences too close
        buffer_distance = 0.1  # degrees (~11 km)
        
        pseudo_absences = []
        max_attempts = n_pseudoabsences * 10
        attempts = 0
        
        while len(pseudo_absences) < n_pseudoabsences and attempts < max_attempts:
            # Random point within Indonesia bounds
            lat = np.random.uniform(self.indonesia_bounds['min_lat'], 
                                  self.indonesia_bounds['max_lat'])
            lng = np.random.uniform(self.indonesia_bounds['min_lng'], 
                                  self.indonesia_bounds['max_lng'])
            
            # Check if point is far enough from presence points
            too_close = False
            for _, presence in presence_points.iterrows():
                distance = np.sqrt((lat - presence['latitude'])**2 + 
                                 (lng - presence['longitude'])**2)
                if distance < buffer_distance:
                    too_close = True
                    break
            
            if not too_close:
                pseudo_absences.append({'latitude': lat, 'longitude': lng})
            
            attempts += 1
        
        return pd.DataFrame(pseudo_absences)
    
    def match_with_environmental_data(self, points):
        """Match occurrence points with environmental data"""
        if self.environmental_data is None:
            return None
        
        # Find nearest environmental data point for each occurrence
        matched_data = []
        
        for _, point in points.iterrows():
            # Calculate distances to all environmental points
            distances = np.sqrt(
                (self.environmental_data['latitude'] - point['latitude'])**2 + 
                (self.environmental_data['longitude'] - point['longitude'])**2
            )
            
            # Find nearest environmental point
            nearest_idx = distances.idxmin()
            
            if distances[nearest_idx] < 0.3:  # Within ~33km (increased tolerance)
                env_point = self.environmental_data.iloc[nearest_idx].copy()
                env_point['presence'] = point['presence']
                env_point['original_lat'] = point['latitude']
                env_point['original_lng'] = point['longitude']
                matched_data.append(env_point)
        
        if matched_data:
            return pd.DataFrame(matched_data)
        else:
            return None
    
    def train_species_model(self, species_name, training_data):
        """Train Random Forest model for a species"""
        print(f"🤖 Training Random Forest model for {species_name}...")
        
        # Prepare features and target
        feature_cols = [col for col in self.env_variables if col in training_data.columns]
        X = training_data[feature_cols].fillna(training_data[feature_cols].mean())
        y = training_data['presence']
        
        print(f"  Features: {len(feature_cols)} environmental variables")
        print(f"  Training samples: {len(X)} ({sum(y)} presences, {len(y)-sum(y)} absences)")
        
        # Check if we have enough data
        if len(X) < 20 or sum(y) < 5:
            print(f"  ⚠️  Insufficient training data")
            return None
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # If stratify fails, try without it
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = rf_model.score(X_train, y_train)
        test_score = rf_model.score(X_test, y_test)
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(rf_model, X, y, cv=min(5, len(X)//5), scoring='roc_auc')
        except:
            cv_scores = np.array([0.5])  # Default if CV fails
        
        # Predictions and metrics
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_score = 0.5
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        model_info = {
            'model': rf_model,
            'feature_cols': feature_cols,
            'feature_importance': feature_importance,
            'train_score': train_score,
            'test_score': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'auc_score': auc_score,
            'n_presences': sum(y),
            'n_absences': len(y) - sum(y),
            'training_data': training_data
        }
        
        print(f"  ✅ Model trained successfully")
        print(f"     Train accuracy: {train_score:.3f}")
        print(f"     Test accuracy: {test_score:.3f}")
        print(f"     CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"     Test AUC: {auc_score:.3f}")
        
        return model_info
    
    def predict_species_distribution(self, species_name, model_info):
        """Predict species distribution across Indonesia"""
        print(f"🗺️  Predicting distribution for {species_name}...")
        
        if self.environmental_data is None:
            self.load_environmental_data()
        
        # Prepare prediction data
        feature_cols = model_info['feature_cols']
        X_pred = self.environmental_data[feature_cols].fillna(
            self.environmental_data[feature_cols].mean()
        )
        
        # Make predictions
        model = model_info['model']
        predictions = model.predict_proba(X_pred)[:, 1]  # Probability of presence
        
        # Create prediction dataframe
        prediction_df = self.environmental_data[['latitude', 'longitude']].copy()
        prediction_df['suitability'] = predictions
        prediction_df['species'] = species_name
        
        print(f"  ✅ Generated {len(prediction_df)} prediction points")
        return prediction_df
    
    def create_feature_importance_chart(self, model_info):
        """Create feature importance visualization"""
        feature_importance = model_info['feature_importance']
        
        # Variable descriptions
        var_descriptions = {
            'bio1': 'Annual Mean Temperature',
            'bio2': 'Mean Diurnal Range',
            'bio3': 'Isothermality',
            'bio4': 'Temperature Seasonality',
            'bio5': 'Max Temperature of Warmest Month',
            'bio6': 'Min Temperature of Coldest Month',
            'bio7': 'Temperature Annual Range',
            'bio8': 'Mean Temperature of Wettest Quarter',
            'bio9': 'Mean Temperature of Driest Quarter',
            'bio10': 'Mean Temperature of Warmest Quarter',
            'bio11': 'Mean Temperature of Coldest Quarter',
            'bio12': 'Annual Precipitation',
            'bio13': 'Precipitation of Wettest Month',
            'bio14': 'Precipitation of Driest Month',
            'bio15': 'Precipitation Seasonality',
            'bio16': 'Precipitation of Wettest Quarter',
            'bio17': 'Precipitation of Driest Quarter',
            'bio18': 'Precipitation of Warmest Quarter',
            'bio19': 'Precipitation of Coldest Quarter',
            'elevation': 'Elevation',
            'slope': 'Slope',
            'aspect': 'Aspect',
            'distance_to_coast': 'Distance to Coast',
            'population_density': 'Population Density'
        }
        
        # Create HTML for feature importance
        importance_html = """
        <div style="padding: 20px; font-family: Arial, sans-serif;">
            <h3 style="color: #2E8B57; margin-bottom: 20px;">Environmental Variable Importance</h3>
            <div style="max-height: 400px; overflow-y: auto;">
        """
        
        max_importance = feature_importance['importance'].max()
        
        for _, row in feature_importance.iterrows():
            importance_pct = (row['importance'] / max_importance) * 100
            var_desc = var_descriptions.get(row['feature'], row['feature'])
            
            importance_html += f"""
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                        <span style="font-size: 12px; font-weight: bold;">{var_desc}</span>
                        <span style="font-size: 11px; color: #666;">{row['importance']:.4f}</span>
                    </div>
                    <div style="background-color: #f0f0f0; height: 15px; border-radius: 7px;">
                        <div style="background-color: #2E8B57; height: 15px; width: {importance_pct}%; border-radius: 7px;"></div>
                    </div>
                </div>
            """
        
        importance_html += """
            </div>
        </div>
        """
        
        return importance_html
    
    def save_model(self, species_name, model_info):
        """Save trained model to disk"""
        model_file = self.results_dir / f"{species_name.replace(' ', '_')}_model.pkl"
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"  💾 Saved model to {model_file}")
    
    def create_comprehensive_map(self, species_models_dict):
        """Create comprehensive map with dropdown species selection"""
        print("🗺️  Creating comprehensive interactive map...")
        
        # Create base map
        center_lat = (self.indonesia_bounds['min_lat'] + self.indonesia_bounds['max_lat']) / 2
        center_lng = (self.indonesia_bounds['min_lng'] + self.indonesia_bounds['max_lng']) / 2
        
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=5,
            tiles='OpenStreetMap'
        )
        
        # Add tile layers
        folium.TileLayer('CartoDB positron', name='Light Mode').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)
        folium.TileLayer('Stamen Terrain', name='Terrain').add_to(m)
        
        # Create species data for JavaScript
        species_data = {}
        
        for species_name, (model_info, predictions) in species_models_dict.items():
            # Prepare heatmap data
            heat_data = []
            presence_points = []
            
            for _, row in predictions.iterrows():
                if row['suitability'] > 0.1:
                    heat_data.append([row['latitude'], row['longitude'], row['suitability']])
            
            # Get presence points
            training_data = model_info['training_data']
            presence_data = training_data[training_data['presence'] == 1]
            
            for _, point in presence_data.iterrows():
                presence_points.append([point['original_lat'], point['original_lng']])
            
            # Feature importance for popup
            feature_importance = model_info['feature_importance'].head(10)
            importance_text = "<br>".join([
                f"{row['feature']}: {row['importance']:.4f}"
                for _, row in feature_importance.iterrows()
            ])
            
            species_data[species_name] = {
                'heat_data': heat_data,
                'presence_points': presence_points,
                'model_performance': {
                    'cv_auc': f"{model_info['cv_mean']:.3f} ± {model_info['cv_std']:.3f}",
                    'test_auc': f"{model_info['auc_score']:.3f}",
                    'test_accuracy': f"{model_info['test_score']:.3f}",
                    'n_presences': model_info['n_presences'],
                    'n_absences': model_info['n_absences']
                },
                'top_features': importance_text
            }
        
        # Create JavaScript for interactive functionality
        species_list = list(species_models_dict.keys())
        
        js_code = f"""
        var speciesData = {json.dumps(species_data)};
        var speciesList = {json.dumps(species_list)};
        var currentHeatmap = null;
        var currentPresenceLayer = null;
        var map = window['{m.get_name()}'];
        
        function updateSpeciesDisplay(speciesName) {{
            // Remove existing layers
            if (currentHeatmap) {{
                map.removeLayer(currentHeatmap);
            }}
            if (currentPresenceLayer) {{
                map.removeLayer(currentPresenceLayer);
            }}
            
            if (speciesName && speciesData[speciesName]) {{
                var data = speciesData[speciesName];
                
                // Add heatmap
                if (data.heat_data.length > 0) {{
                    currentHeatmap = L.heatLayer(data.heat_data, {{
                        radius: 8,
                        blur: 10,
                        minOpacity: 0.2,
                        gradient: {{0.4: 'blue', 0.6: 'cyan', 0.7: 'lime', 0.8: 'yellow', 1.0: 'red'}}
                    }}).addTo(map);
                }}
                
                // Add presence points
                currentPresenceLayer = L.layerGroup();
                data.presence_points.forEach(function(point) {{
                    L.circleMarker([point[0], point[1]], {{
                        radius: 4,
                        color: 'green',
                        weight: 2,
                        fillColor: 'lightgreen',
                        fillOpacity: 0.8
                    }}).bindPopup('Observed ' + speciesName).addTo(currentPresenceLayer);
                }});
                currentPresenceLayer.addTo(map);
                
                // Update info panel
                updateInfoPanel(speciesName, data);
            }}
        }}
        
        function updateInfoPanel(speciesName, data) {{
            var infoDiv = document.getElementById('speciesInfo');
            if (infoDiv) {{
                infoDiv.innerHTML = `
                    <h4 style="margin: 0 0 10px 0; color: #2E8B57;">${{speciesName}}</h4>
                    <div style="margin-bottom: 8px;">
                        <b>Model Performance:</b><br>
                        CV AUC: ${{data.model_performance.cv_auc}}<br>
                        Test AUC: ${{data.model_performance.test_auc}}<br>
                        Test Accuracy: ${{data.model_performance.test_accuracy}}
                    </div>
                    <div style="margin-bottom: 8px;">
                        <b>Training Data:</b><br>
                        Presences: ${{data.model_performance.n_presences}}<br>
                        Absences: ${{data.model_performance.n_absences}}
                    </div>
                    <details style="margin-top: 10px;">
                        <summary style="cursor: pointer; font-weight: bold;">Top Environmental Factors</summary>
                        <div style="font-size: 10px; margin-top: 5px; max-height: 150px; overflow-y: auto;">
                            ${{data.top_features}}
                        </div>
                    </details>
                `;
            }}
        }}
        
        // Initialize dropdown
        setTimeout(function() {{
            var controlDiv = document.createElement('div');
            controlDiv.style.cssText = `
                position: fixed; top: 10px; left: 50px; z-index: 9999;
                background-color: white; padding: 10px; border-radius: 5px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            `;
            
            var select = document.createElement('select');
            select.style.cssText = 'width: 250px; font-size: 12px; padding: 5px;';
            select.innerHTML = '<option value="">Select a species...</option>';
            
            speciesList.forEach(function(species) {{
                var option = document.createElement('option');
                option.value = species;
                option.textContent = species;
                select.appendChild(option);
            }});
            
            select.addEventListener('change', function() {{
                updateSpeciesDisplay(this.value);
            }});
            
            controlDiv.appendChild(select);
            document.body.appendChild(controlDiv);
            
            // Create info panel
            var infoDiv = document.createElement('div');
            infoDiv.id = 'speciesInfo';
            infoDiv.style.cssText = `
                position: fixed; top: 10px; right: 10px; width: 320px;
                background-color: white; border: 2px solid grey; z-index: 9999;
                font-size: 12px; padding: 15px; border-radius: 5px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2); max-height: 400px;
                overflow-y: auto;
            `;
            infoDiv.innerHTML = '<p style="color: #666;">Select a species from the dropdown to view its distribution model.</p>';
            document.body.appendChild(infoDiv);
        }}, 1000);
        """
        
        # Add JavaScript to map
        m.get_root().html.add_child(folium.Element(f"<script>{js_code}</script>"))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        return m
    
    def run_modeling_workflow(self, max_species=10, min_occurrences=20):
        """Run complete species distribution modeling workflow"""
        print("🚀 Starting Species Distribution Modeling Workflow")
        print("="*60)
        
        # Load environmental data
        env_data = self.load_environmental_data()
        if env_data is None or len(env_data) == 0:
            print("❌ No environmental data available. Please run environmental data download first.")
            return None
        
        # Get species list
        species_list = self.get_species_list(min_occurrences)
        
        if not species_list:
            print("❌ No species found with sufficient occurrences")
            return None
        
        # Limit to max_species for demonstration
        species_to_model = species_list[:max_species]
        print(f"\n🎯 Modeling {len(species_to_model)} species:")
        for i, species in enumerate(species_to_model, 1):
            print(f"  {i:2d}. {species}")
        
        # Model each species
        species_models_dict = {}
        
        for i, species_name in enumerate(species_to_model, 1):
            print(f"\n{'='*60}")
            print(f"MODELING SPECIES {i}/{len(species_to_model)}: {species_name}")
            print("="*60)
            
            try:
                # Prepare training data
                training_data = self.prepare_species_data(species_name)
                if training_data is None:
                    print(f"❌ Skipping {species_name} - insufficient data")
                    continue
                
                # Train model
                model_info = self.train_species_model(species_name, training_data)
                if model_info is None:
                    print(f"❌ Failed to train model for {species_name}")
                    continue
                
                # Make predictions
                predictions = self.predict_species_distribution(species_name, model_info)
                
                # Save individual model
                self.save_model(species_name, model_info)
                
                # Store for comprehensive map
                species_models_dict[species_name] = (model_info, predictions)
                
                print(f"✅ Successfully modeled {species_name}")
                
            except Exception as e:
                print(f"❌ Error modeling {species_name}: {e}")
                continue
        
        if not species_models_dict:
            print("❌ No models were successfully trained")
            return None
        
        print(f"\n{'='*60}")
        print(f"CREATING COMPREHENSIVE VISUALIZATION")
        print("="*60)
        
        # Create comprehensive interactive map
        comprehensive_map = self.create_comprehensive_map(species_models_dict)
        
        # Save map
        map_filename = self.results_dir / "species_distribution_models.html"
        comprehensive_map.save(str(map_filename))
        
        print(f"\n🎉 MODELING WORKFLOW COMPLETE!")
        print("="*60)
        print(f"Successfully modeled: {len(species_models_dict)} species")
        print(f"Interactive map saved: {map_filename}")
        print(f"Individual models saved in: {self.results_dir}")
        
        # Print summary statistics
        print(f"\nModel Performance Summary:")
        for species_name, (model_info, _) in species_models_dict.items():
            print(f"  {species_name:<40} AUC: {model_info['auc_score']:.3f}")
        
        return species_models_dict, str(map_filename)

def main():
    """Main function to run species distribution modeling"""
    print("🌍 Indonesian Species Distribution Modeling")
    print("Using Random Forest with Environmental Predictors")
    print("="*60)
    
    # Initialize modeler
    modeler = SpeciesDistributionModeler()
    
    # Configuration
    max_species = 15  # Number of species to model
    min_occurrences = 25  # Minimum occurrences required
    
    print(f"Configuration:")
    print(f"  - Maximum species to model: {max_species}")
    print(f"  - Minimum occurrences per species: {min_occurrences}")
    print(f"  - Model type: Random Forest")
    print(f"  - Pseudo-absence strategy: Random sampling with buffer")
    
    # Run modeling workflow
    result = modeler.run_modeling_workflow(
        max_species=max_species,
        min_occurrences=min_occurrences
    )
    
    if result:
        models_dict, map_file = result
        print(f"\n✅ Successfully created species distribution models!")
        print(f"📊 Models trained: {len(models_dict)}")
        print(f"🗺️  Interactive map: {map_file}")
        print(f"\n🔍 Open {map_file} in your browser to explore the results!")
        print(f"\n📋 Usage Instructions:")
        print(f"  1. Open {map_file} in your web browser")
        print(f"  2. Use the dropdown menu to select different species")
        print(f"  3. View habitat suitability heatmap (blue=low, red=high)")
        print(f"  4. Green circles show actual observation points")
        print(f"  5. Right panel shows model performance and feature importance")
        print(f"  6. Toggle between different map layers using the layer control")
        
        print(f"\n🔬 Model Insights:")
        avg_auc = np.mean([info[0]['auc_score'] for info in models_dict.values()])
        print(f"  - Average model AUC: {avg_auc:.3f}")
        print(f"  - Environmental variables used: {len(modeler.env_variables)}")
        print(f"  - Resolution: ~28km environmental grid")
        
    else:
        print("❌ Species distribution modeling failed")
        print("💡 Make sure you have:")
        print("  1. Run the environmental data processor first")
        print("  2. Have sufficient species occurrence data in your database")
        print("  3. Installed required packages: scikit-learn, folium")

if __name__ == "__main__":
    main()