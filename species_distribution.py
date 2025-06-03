import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.neighbors import NearestNeighbors
import folium
from folium import plugins
import json
import pickle
import struct
from pathlib import Path
from collections import Counter
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

class FixedSpeciesDistributionModeler:
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
        
        # Initialize data containers
        self.environmental_data = None
        self.env_nn_model = None
        self.species_models = {}
        
        # Environmental variables for modeling
        self.env_variables = [
            'bio1', 'bio2', 'bio3', 'bio4', 'bio5', 'bio6', 'bio7', 'bio8',
            'bio9', 'bio10', 'bio11', 'bio12', 'bio13', 'bio14', 'bio15',
            'bio16', 'bio17', 'bio18', 'bio19', 'elevation', 'slope',
            'aspect', 'distance_to_coast', 'population_density'
        ]
        
    def convert_blob_to_float(self, blob_data):
        """Convert binary blob data to float using struct unpacking"""
        if blob_data is None:
            return np.nan
        
        try:
            # Try to unpack as float (4 bytes, little endian)
            if len(blob_data) == 4:
                return struct.unpack('<f', blob_data)[0]
            # Try to unpack as double (8 bytes, little endian)
            elif len(blob_data) == 8:
                return struct.unpack('<d', blob_data)[0]
            else:
                return np.nan
        except (struct.error, TypeError):
            return np.nan
    
    def fix_environmental_data_in_db(self):
        """Fix the environmental data by converting blobs to proper floats"""
        print("🔧 Fixing environmental data in database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # First, get all data with blob columns
        cursor.execute("SELECT * FROM environmental_data")
        rows = cursor.fetchall()
        
        if not rows:
            print("❌ No environmental data found in database")
            conn.close()
            return False
        
        # Get column names
        cursor.execute("PRAGMA table_info(environmental_data)")
        columns_info = cursor.fetchall()
        column_names = [row[1] for row in columns_info]
        
        print(f"📊 Processing {len(rows)} rows with {len(column_names)} columns...")
        
        # Process each row
        fixed_rows = []
        for row in rows:
            fixed_row = list(row)
            
            for i, (col_name, value) in enumerate(zip(column_names, row)):
                if col_name in self.env_variables and isinstance(value, bytes):
                    # Convert blob to float
                    fixed_value = self.convert_blob_to_float(value)
                    fixed_row[i] = fixed_value
            
            fixed_rows.append(tuple(fixed_row))
        
        # Clear and rebuild the table with correct data
        print("🗑️  Clearing old data...")
        cursor.execute("DELETE FROM environmental_data")
        
        # Re-insert fixed data
        print("💾 Inserting fixed data...")
        placeholders = ', '.join(['?' for _ in column_names])
        cursor.executemany(
            f"INSERT INTO environmental_data ({', '.join(column_names)}) VALUES ({placeholders})",
            fixed_rows
        )
        
        conn.commit()
        conn.close()
        
        print("✅ Environmental data fixed!")
        return True
        
    def load_and_prepare_environmental_data(self):
        """Load environmental data and prepare for fast matching"""
        print("📊 Loading and preparing environmental data...")
        
        # First, try to fix the data if it's corrupted
        print("🔍 Checking for binary data corruption...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if environmental_data table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='environmental_data'")
        if not cursor.fetchone():
            print("❌ Environmental data table not found! Please run download_environmental.py first.")
            conn.close()
            return False
        
        # Check if we have blob data in bioclimatic variables
        cursor.execute("SELECT bio1 FROM environmental_data LIMIT 1")
        sample = cursor.fetchone()
        
        if sample and isinstance(sample[0], bytes):
            print("🔧 Detected binary blob data - fixing...")
            conn.close()
            if not self.fix_environmental_data_in_db():
                return False
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
        
        # Check available columns
        cursor.execute("PRAGMA table_info(environmental_data)")
        available_columns = [row[1] for row in cursor.fetchall()]
        
        # Update env_variables to only include available columns
        self.env_variables = [col for col in self.env_variables if col in available_columns]
        
        if len(self.env_variables) < 3:
            print(f"❌ Insufficient environmental variables available! Found: {available_columns}")
            conn.close()
            return False
        
        print(f"🔍 Found {len(self.env_variables)} environmental variables")
        
        # Load environmental data efficiently
        env_vars_str = ', '.join(self.env_variables)
        query = f"""
        SELECT latitude, longitude, {env_vars_str}
        FROM environmental_data
        WHERE latitude IS NOT NULL 
        AND longitude IS NOT NULL
        """
        
        try:
            self.environmental_data = pd.read_sql_query(query, conn)
            conn.close()
        except Exception as e:
            print(f"❌ Error loading environmental data: {e}")
            conn.close()
            return False
        
        if len(self.environmental_data) == 0:
            print("❌ No environmental data found!")
            return False
        
        print(f"📊 Loaded {len(self.environmental_data)} raw environmental records")
        
        # Convert all environmental variables to numeric
        print("🧹 Converting data types...")
        for var in self.env_variables:
            if var in self.environmental_data.columns:
                self.environmental_data[var] = pd.to_numeric(
                    self.environmental_data[var], 
                    errors='coerce'
                )
        
        # Check for missing data and data quality
        print("📊 Data quality check:")
        valid_vars = []
        for var in self.env_variables:
            if var in self.environmental_data.columns:
                valid_count = self.environmental_data[var].notna().sum()
                valid_pct = (valid_count / len(self.environmental_data)) * 100
                print(f"   {var}: {valid_count}/{len(self.environmental_data)} ({valid_pct:.1f}%) valid")
                
                if valid_pct > 10:  # Keep variables with >10% valid data
                    valid_vars.append(var)
                else:
                    print(f"   ❌ Dropping {var} - insufficient valid data")
        
        self.env_variables = valid_vars
        
        if len(self.env_variables) < 3:
            print(f"❌ Insufficient valid environmental variables ({len(self.env_variables)})")
            return False
        
        # Remove rows with too many missing values
        min_vars_required = max(3, len(self.env_variables) // 3)
        initial_count = len(self.environmental_data)
        
        self.environmental_data = self.environmental_data.dropna(
            subset=self.env_variables, 
            thresh=min_vars_required
        )
        
        removed_count = initial_count - len(self.environmental_data)
        if removed_count > 0:
            print(f"   🗑️  Removed {removed_count} rows with insufficient data")
        
        if len(self.environmental_data) < 50:
            print(f"❌ Insufficient environmental data ({len(self.environmental_data)} points)")
            return False
        
        # Fill missing values with median
        print("🔧 Filling missing values...")
        for var in self.env_variables:
            if var in self.environmental_data.columns:
                missing_count = self.environmental_data[var].isna().sum()
                if missing_count > 0:
                    median_val = self.environmental_data[var].median()
                    self.environmental_data[var] = self.environmental_data[var].fillna(median_val)
                    print(f"   📊 Filled {missing_count} missing values in {var}")
        
        # Create spatial index
        print("🔍 Creating spatial index...")
        try:
            coords = self.environmental_data[['latitude', 'longitude']].values
            self.env_nn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='haversine')
            self.env_nn_model.fit(np.radians(coords))
        except Exception as e:
            print(f"❌ Error creating spatial index: {e}")
            return False
        
        print(f"✅ Successfully prepared {len(self.environmental_data)} environmental points")
        print(f"📊 Using {len(self.env_variables)} variables: {', '.join(self.env_variables)}")
        
        return True
        
    def get_species_list_optimized(self, min_occurrences=20, max_species=None):
        """Get list of species with sufficient occurrence records"""
        print(f"🔍 Finding species with at least {min_occurrences} occurrences...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Check if occurrences table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='occurrences'")
        if not cursor.fetchone():
            print("❌ Occurrences table not found! Please run download_records.py first.")
            conn.close()
            return pd.DataFrame()
        
        limit_clause = f"LIMIT {max_species}" if max_species else ""
        
        query = f"""
        SELECT species, COUNT(*) as occurrence_count
        FROM occurrences 
        WHERE species IS NOT NULL 
        AND decimalLatitude BETWEEN {self.indonesia_bounds['min_lat']} AND {self.indonesia_bounds['max_lat']}
        AND decimalLongitude BETWEEN {self.indonesia_bounds['min_lng']} AND {self.indonesia_bounds['max_lng']}
        GROUP BY species
        HAVING COUNT(*) >= {min_occurrences}
        ORDER BY COUNT(*) DESC
        {limit_clause}
        """
        
        try:
            species_counts = pd.read_sql_query(query, conn)
            conn.close()
        except Exception as e:
            print(f"❌ Error querying species: {e}")
            conn.close()
            return pd.DataFrame()
        
        print(f"✅ Found {len(species_counts)} species with ≥{min_occurrences} occurrences")
        return species_counts
    
    def load_all_occurrence_data(self, species_list):
        """Load all occurrence data for selected species at once"""
        print("📋 Loading all occurrence data...")
        
        if not species_list:
            return pd.DataFrame()
        
        species_str = "', '".join(species_list)
        
        conn = sqlite3.connect(self.db_path)
        query = f"""
        SELECT species, decimalLatitude as latitude, decimalLongitude as longitude
        FROM occurrences 
        WHERE species IN ('{species_str}')
        AND decimalLatitude BETWEEN {self.indonesia_bounds['min_lat']} AND {self.indonesia_bounds['max_lat']}
        AND decimalLongitude BETWEEN {self.indonesia_bounds['min_lng']} AND {self.indonesia_bounds['max_lng']}
        """
        
        try:
            all_occurrences = pd.read_sql_query(query, conn)
            conn.close()
        except Exception as e:
            print(f"❌ Error loading occurrences: {e}")
            conn.close()
            return pd.DataFrame()
        
        # Remove duplicates within species
        all_occurrences = all_occurrences.drop_duplicates(subset=['species', 'latitude', 'longitude'])
        
        print(f"✅ Loaded {len(all_occurrences)} unique occurrence records")
        return all_occurrences
    
    def fast_environmental_matching(self, occurrence_points):
        """Fast environmental matching using nearest neighbors"""
        if len(occurrence_points) == 0:
            return pd.DataFrame()
        
        # Convert to radians for haversine distance
        coords_rad = np.radians(occurrence_points[['latitude', 'longitude']].values)
        
        # Find nearest environmental points
        distances, indices = self.env_nn_model.kneighbors(coords_rad)
        
        # Convert distances back to degrees (approximately)
        distances_deg = distances[:, 0] * 180 / np.pi
        
        # Filter points that are too far (>0.5 degrees ~55km)
        valid_mask = distances_deg < 0.5
        
        if not np.any(valid_mask):
            return pd.DataFrame()
        
        # Get matched environmental data
        valid_indices = indices[valid_mask, 0]
        matched_env_data = self.environmental_data.iloc[valid_indices].copy().reset_index(drop=True)
        
        # Add original occurrence info INCLUDING the presence column
        valid_occurrences = occurrence_points[valid_mask].reset_index(drop=True)
        for col in valid_occurrences.columns:
            matched_env_data[col] = valid_occurrences[col].values
        
        return matched_env_data
    
    def generate_pseudo_absences_vectorized(self, presence_points, n_pseudoabsences, buffer_distance=0.1):
        """Generate pseudo-absences using vectorized operations"""
        np.random.seed(42)
        
        # Pre-generate many random points
        n_candidates = n_pseudoabsences * 5
        
        candidate_lats = np.random.uniform(
            self.indonesia_bounds['min_lat'], 
            self.indonesia_bounds['max_lat'], 
            n_candidates
        )
        candidate_lngs = np.random.uniform(
            self.indonesia_bounds['min_lng'], 
            self.indonesia_bounds['max_lng'], 
            n_candidates
        )
        
        # Vectorized distance calculation to all presence points
        presence_coords = presence_points[['latitude', 'longitude']].values
        
        valid_absences = []
        for i in range(n_candidates):
            if len(valid_absences) >= n_pseudoabsences:
                break
                
            candidate = np.array([candidate_lats[i], candidate_lngs[i]])
            
            # Calculate distances to all presence points
            distances = np.sqrt(np.sum((presence_coords - candidate)**2, axis=1))
            
            # Check if far enough from all presence points
            if np.min(distances) >= buffer_distance:
                valid_absences.append({
                    'latitude': candidate_lats[i], 
                    'longitude': candidate_lngs[i]
                })
        
        return pd.DataFrame(valid_absences[:n_pseudoabsences])
    
    def prepare_training_data_batch(self, all_occurrences, species_name):
        """Prepare training data for a species using batch processing"""
        # Get presence points for this species
        species_presences = all_occurrences[
            all_occurrences['species'] == species_name
        ][['latitude', 'longitude']].drop_duplicates()
        
        if len(species_presences) < 10:
            return None
        
        # Generate pseudo-absences
        n_pseudoabsences = min(len(species_presences) * 2, 300)
        pseudo_absences = self.generate_pseudo_absences_vectorized(
            species_presences, n_pseudoabsences
        )
        
        if len(pseudo_absences) < 10:
            return None
        
        # Combine presence and absence points
        species_presences['presence'] = 1
        pseudo_absences['presence'] = 0
        
        all_points = pd.concat([species_presences, pseudo_absences], ignore_index=True)
        
        # Fast environmental matching
        training_data = self.fast_environmental_matching(all_points)
        
        if len(training_data) < 20:
            return None
        
        return training_data
    
    def train_optimized_model(self, species_name, training_data):
        """Train an optimized Random Forest model"""
        # Check if presence column exists
        if 'presence' not in training_data.columns:
            print(f"     ❌ Missing 'presence' column. Available columns: {list(training_data.columns)}")
            return None
        
        # Prepare features
        feature_cols = [col for col in self.env_variables if col in training_data.columns]
        if len(feature_cols) == 0:
            print(f"     ❌ No environmental feature columns found")
            return None
            
        X = training_data[feature_cols]
        y = training_data['presence']
        
        if len(X) < 20 or sum(y) < 5:
            print(f"     ❌ Insufficient data: {len(X)} total points, {sum(y)} presences")
            return None
        
        # Use smaller, faster Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            max_features='sqrt'
        )
        
        # Simple train-test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Train model
        rf_model.fit(X_train, y_train)
        
        # Quick evaluation
        test_score = rf_model.score(X_test, y_test)
        
        try:
            y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
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
            'test_score': test_score,
            'auc_score': auc_score,
            'n_presences': sum(y),
            'n_absences': len(y) - sum(y),
            'n_training_points': len(training_data),
            'training_data': training_data  # Store for visualization
        }
        
        return model_info
    
    def predict_distribution_fast(self, model_info):
        """Fast prediction across environmental grid"""
        feature_cols = model_info['feature_cols']
        X_pred = self.environmental_data[feature_cols]
        
        # Make predictions
        model = model_info['model']
        predictions = model.predict_proba(X_pred)[:, 1]
        
        # Create prediction dataframe
        prediction_df = self.environmental_data[['latitude', 'longitude']].copy()
        prediction_df['suitability'] = predictions
        
        return prediction_df
    
    def create_robust_interactive_map(self, species_models_dict):
        """Create a robust interactive map with better error handling and debugging"""
        print("🗺️  Creating robust interactive map...")
        
        if not species_models_dict:
            print("❌ No species models provided")
            return None
        
        # Create base map
        m = folium.Map(
            location=[-2.5, 118.0],
            zoom_start=5,
            tiles='OpenStreetMap'
        )
        
        # Add tile layers
        folium.TileLayer('CartoDB positron', name='Light').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark').add_to(m)
        
        # Process species data more robustly
        processed_species_data = {}
        
        for species_name, (model_info, predictions) in species_models_dict.items():
            print(f"     Processing {species_name}...")
            
            try:
                # Validate prediction data
                if predictions is None or len(predictions) == 0:
                    print(f"       ❌ No prediction data for {species_name}")
                    continue
                
                if 'suitability' not in predictions.columns:
                    print(f"       ❌ No suitability column for {species_name}")
                    continue
                
                # Clean and validate data
                valid_predictions = predictions.dropna(subset=['latitude', 'longitude', 'suitability'])
                
                if len(valid_predictions) == 0:
                    print(f"       ❌ No valid predictions for {species_name}")
                    continue
                
                print(f"       Valid predictions: {len(valid_predictions)}")
                print(f"       Suitability range: {valid_predictions['suitability'].min():.3f} - {valid_predictions['suitability'].max():.3f}")
                
                # Sample for performance (every 5th point)
                sampled = valid_predictions.iloc[::5]
                
                # Create different categories of points
                high_suit = sampled[sampled['suitability'] > 0.7]
                med_suit = sampled[(sampled['suitability'] > 0.4) & (sampled['suitability'] <= 0.7)]
                low_suit = sampled[(sampled['suitability'] > 0.1) & (sampled['suitability'] <= 0.4)]
                
                print(f"       Categories - High: {len(high_suit)}, Medium: {len(med_suit)}, Low: {len(low_suit)}")
                
                # Convert to simple lists for JavaScript
                high_points = [[float(row['latitude']), float(row['longitude']), float(row['suitability'])] 
                              for _, row in high_suit.iterrows()]
                med_points = [[float(row['latitude']), float(row['longitude']), float(row['suitability'])] 
                             for _, row in med_suit.iterrows()]
                low_points = [[float(row['latitude']), float(row['longitude']), float(row['suitability'])] 
                             for _, row in low_suit.iterrows()]
                
                # Get presence points from training data
                presence_points = []
                if 'training_data' in model_info:
                    training_data = model_info['training_data']
                    if training_data is not None and 'presence' in training_data.columns:
                        presence_data = training_data[training_data['presence'] == 1]
                        
                        # Limit to 30 points for performance
                        if len(presence_data) > 30:
                            presence_data = presence_data.sample(n=30, random_state=42)
                        
                        for _, row in presence_data.iterrows():
                            try:
                                lat = float(row['latitude'])
                                lng = float(row['longitude'])
                                presence_points.append([lat, lng])
                            except (ValueError, TypeError, KeyError):
                                continue
                
                print(f"       Presence points: {len(presence_points)}")
                
                # Feature importance
                feature_importance = model_info.get('feature_importance', pd.DataFrame())
                if not feature_importance.empty:
                    top_features = feature_importance.head(5)
                    importance_text = "<br>".join([
                        f"{row['feature']}: {row['importance']:.3f}"
                        for _, row in top_features.iterrows()
                    ])
                else:
                    importance_text = "No feature importance data available"
                
                # Store processed data
                processed_species_data[species_name] = {
                    'high_points': high_points[:100],  # Limit for performance
                    'med_points': med_points[:150],
                    'low_points': low_points[:200],
                    'presence_points': presence_points,
                    'model_performance': {
                        'test_accuracy': f"{model_info.get('test_score', 0):.3f}",
                        'auc_score': f"{model_info.get('auc_score', 0):.3f}",
                        'n_presences': int(model_info.get('n_presences', 0)),
                        'n_training_points': int(model_info.get('n_training_points', 0))
                    },
                    'top_features': importance_text
                }
                
                print(f"       ✅ Successfully processed {species_name}")
                
            except Exception as e:
                print(f"       ❌ Error processing {species_name}: {str(e)}")
                continue
        
        if not processed_species_data:
            print("❌ No species data could be processed")
            return None
        
        print(f"✅ Successfully processed {len(processed_species_data)} species")
        
        # Create JavaScript for map interaction
        species_list = list(processed_species_data.keys())
        
        # Get the map variable name that Folium generates
        map_var_name = m.get_name()
        
        # More robust JavaScript implementation
        js_code = f"""
    var speciesData = {json.dumps(processed_species_data, ensure_ascii=False, indent=2)};
    var speciesList = {json.dumps(species_list, ensure_ascii=False)};
    var currentLayers = [];
    var mapVarName = '{map_var_name}';
    var map = window[mapVarName];

    console.log('Map variable name:', mapVarName);
    console.log('Map object:', map);
    console.log('Species data loaded:', Object.keys(speciesData).length, 'species');

    function findMap() {{
        // Try multiple ways to find the map object
        if (window[mapVarName]) {{
            return window[mapVarName];
        }}
        
        // Try to find any Leaflet map in window
        for (var key in window) {{
            if (window[key] && window[key]._container && window[key].addLayer) {{
                console.log('Found map via fallback:', key);
                return window[key];
            }}
        }}
        
        // Try to find via Leaflet's global maps
        if (window.L && window.L.Map) {{
            var maps = document.querySelectorAll('.folium-map');
            if (maps.length > 0) {{
                var mapDiv = maps[0];
                for (var key in window) {{
                    if (window[key] && window[key]._container === mapDiv) {{
                        console.log('Found map via container match:', key);
                        return window[key];
                    }}
                }}
            }}
        }}
        
        console.error('Could not find map object');
        return null;
    }}

    function clearAllLayers() {{
        try {{
            var currentMap = findMap();
            if (!currentMap) {{
                console.error('Cannot clear layers: map not found');
                return;
            }}
            
            currentLayers.forEach(function(layer) {{
                try {{
                    if (currentMap.hasLayer && currentMap.hasLayer(layer)) {{
                        currentMap.removeLayer(layer);
                    }}
                }} catch (e) {{
                    console.error('Error removing layer:', e);
                }}
            }});
            currentLayers = [];
            console.log('Cleared all layers');
        }} catch (e) {{
            console.error('Error clearing layers:', e);
        }}
    }}

    function addMarkersForSpecies(speciesName) {{
        console.log('Adding markers for:', speciesName);
        
        var currentMap = findMap();
        if (!currentMap) {{
            console.error('Cannot add markers: map not found');
            return;
        }}
        
        if (!speciesData[speciesName]) {{
            console.error('No data found for species:', speciesName);
            return;
        }}
        
        var data = speciesData[speciesName];
        console.log('Species data:', data);
        
        try {{
            // High suitability markers (red)
            if (data.high_points && data.high_points.length > 0) {{
                console.log('Adding', data.high_points.length, 'high suitability points');
                data.high_points.forEach(function(point) {{
                    try {{
                        var marker = L.circleMarker([point[0], point[1]], {{
                            radius: 8,
                            color: 'darkred',
                            weight: 1,
                            fillColor: 'red',
                            fillOpacity: 0.8
                        }}).bindTooltip('High suitability: ' + point[2].toFixed(3));
                        
                        marker.addTo(currentMap);
                        currentLayers.push(marker);
                    }} catch (e) {{
                        console.error('Error adding high point:', e, point);
                    }}
                }});
            }}
            
            // Medium suitability markers (orange)
            if (data.med_points && data.med_points.length > 0) {{
                console.log('Adding', data.med_points.length, 'medium suitability points');
                data.med_points.forEach(function(point) {{
                    try {{
                        var marker = L.circleMarker([point[0], point[1]], {{
                            radius: 6,
                            color: 'darkorange',
                            weight: 1,
                            fillColor: 'orange',
                            fillOpacity: 0.7
                        }}).bindTooltip('Medium suitability: ' + point[2].toFixed(3));
                        
                        marker.addTo(currentMap);
                        currentLayers.push(marker);
                    }} catch (e) {{
                        console.error('Error adding medium point:', e, point);
                    }}
                }});
            }}
            
            // Low suitability markers (yellow)
            if (data.low_points && data.low_points.length > 0) {{
                console.log('Adding', data.low_points.length, 'low suitability points');
                data.low_points.forEach(function(point) {{
                    try {{
                        var marker = L.circleMarker([point[0], point[1]], {{
                            radius: 4,
                            color: 'goldenrod',
                            weight: 1,
                            fillColor: 'yellow',
                            fillOpacity: 0.6
                        }}).bindTooltip('Low suitability: ' + point[2].toFixed(3));
                        
                        marker.addTo(currentMap);
                        currentLayers.push(marker);
                    }} catch (e) {{
                        console.error('Error adding low point:', e, point);
                    }}
                }});
            }}
            
            // Presence points (green)
            if (data.presence_points && data.presence_points.length > 0) {{
                console.log('Adding', data.presence_points.length, 'presence points');
                data.presence_points.forEach(function(point) {{
                    try {{
                        var marker = L.circleMarker([point[0], point[1]], {{
                            radius: 6,
                            color: 'darkgreen',
                            weight: 2,
                            fillColor: 'lightgreen',
                            fillOpacity: 0.9
                        }}).bindTooltip('Observed: ' + speciesName);
                        
                        marker.addTo(currentMap);
                        currentLayers.push(marker);
                    }} catch (e) {{
                        console.error('Error adding presence point:', e, point);
                    }}
                }});
            }}
            
            console.log('Total layers added:', currentLayers.length);
            updateInfoPanel(speciesName, data);
            
        }} catch (e) {{
            console.error('Error adding markers:', e);
        }}
    }}

    function updateInfoPanel(speciesName, data) {{
        var infoDiv = document.getElementById('speciesInfo');
        if (infoDiv && data) {{
            var truncatedName = speciesName.length > 35 ? speciesName.substring(0, 35) + '...' : speciesName;
            infoDiv.innerHTML = `
                <h4 style="margin: 0 0 10px 0; color: #2E8B57; font-size: 13px;">${{truncatedName}}</h4>
                <div style="margin-bottom: 8px; font-size: 11px;">
                    <b>Model Performance:</b><br>
                    Accuracy: ${{data.model_performance.test_accuracy}}<br>
                    AUC Score: ${{data.model_performance.auc_score}}<br>
                    Training Points: ${{data.model_performance.n_training_points}}<br>
                    Presences: ${{data.model_performance.n_presences}}
                </div>
                <div style="margin-bottom: 8px; font-size: 10px; color: #666;">
                    <b>Habitat Suitability:</b><br>
                    High (>0.7): ${{data.high_points.length}} points<br>
                    Medium (0.4-0.7): ${{data.med_points.length}} points<br>
                    Low (0.1-0.4): ${{data.low_points.length}} points<br>
                    Observations: ${{data.presence_points.length}} points
                </div>
                <details style="margin-top: 8px;">
                    <summary style="cursor: pointer; font-weight: bold; font-size: 11px;">Environmental Factors</summary>
                    <div style="font-size: 9px; margin-top: 5px; max-height: 120px; overflow-y: auto;">
                        ${{data.top_features}}
                    </div>
                </details>
            `;
        }}
    }}

    function calculateOverallFeatureImportance() {{
        var aggregatedFeatures = {{}};
        var speciesCount = 0;
        
        Object.keys(speciesData).forEach(function(speciesName) {{
            speciesCount++;
            var features = speciesData[speciesName].top_features.split('<br>');
            features.forEach(function(feature) {{
                if (feature.trim()) {{
                    var parts = feature.split(':');
                    if (parts.length === 2) {{
                        var featureName = parts[0].trim();
                        var importance = parseFloat(parts[1].trim());
                        
                        if (!aggregatedFeatures[featureName]) {{
                            aggregatedFeatures[featureName] = {{sum: 0, count: 0, avg: 0}};
                        }}
                        aggregatedFeatures[featureName].sum += importance;
                        aggregatedFeatures[featureName].count++;
                    }}
                }}
            }});
        }});
        
        // Calculate averages and sort
        var sortedFeatures = Object.keys(aggregatedFeatures).map(function(feature) {{
            var avg = aggregatedFeatures[feature].sum / aggregatedFeatures[feature].count;
            return {{
                feature: feature,
                avgImportance: avg,
                speciesCount: aggregatedFeatures[feature].count,
                totalSpecies: speciesCount
            }};
        }}).sort(function(a, b) {{
            return b.avgImportance - a.avgImportance;
        }});
        
        return sortedFeatures;
    }}

    function showFeatureImportanceSummary() {{
        var summaryFeatures = calculateOverallFeatureImportance();
        
        var summaryHtml = `
            <div style="padding: 20px; max-height: 500px; overflow-y: auto;">
                <h3 style="color: #2E8B57; margin-bottom: 15px;">Overall Feature Importance Summary</h3>
                <p style="font-size: 12px; color: #666; margin-bottom: 15px;">
                    Aggregated across all ${{Object.keys(speciesData).length}} species models
                </p>
                <table style="width: 100%; border-collapse: collapse; font-size: 11px;">
                    <thead>
                        <tr style="background-color: #f0f8ff; font-weight: bold;">
                            <th style="border: 1px solid #ccc; padding: 8px; text-align: left;">Rank</th>
                            <th style="border: 1px solid #ccc; padding: 8px; text-align: left;">Environmental Variable</th>
                            <th style="border: 1px solid #ccc; padding: 8px; text-align: right;">Avg Importance</th>
                            <th style="border: 1px solid #ccc; padding: 8px; text-align: right;">Used in Models</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        summaryFeatures.forEach(function(item, index) {{
            var percentage = (item.speciesCount / item.totalSpecies * 100).toFixed(1);
            var rowColor = index % 2 === 0 ? '#f9f9f9' : 'white';
            var importanceColor = item.avgImportance > 0.1 ? '#2E8B57' : '#666';
            
            summaryHtml += `
                <tr style="background-color: ${{rowColor}};">
                    <td style="border: 1px solid #ccc; padding: 6px; text-align: center; font-weight: bold;">${{index + 1}}</td>
                    <td style="border: 1px solid #ccc; padding: 6px; font-weight: bold;">${{item.feature}}</td>
                    <td style="border: 1px solid #ccc; padding: 6px; text-align: right; color: ${{importanceColor}}; font-weight: bold;">${{item.avgImportance.toFixed(4)}}</td>
                    <td style="border: 1px solid #ccc; padding: 6px; text-align: right;">${{item.speciesCount}}/${{item.totalSpecies}} (${{percentage}}%)</td>
                </tr>
            `;
        }});
        
        summaryHtml += `
                    </tbody>
                </table>
                <div style="margin-top: 15px; padding: 10px; background-color: #f5f5dc; border-radius: 5px; font-size: 10px;">
                    <strong>Variable Descriptions:</strong><br>
                    • <strong>bio1-19:</strong> WorldClim bioclimatic variables (temperature, precipitation patterns)<br>
                    • <strong>elevation:</strong> Altitude above sea level (meters)<br>
                    • <strong>slope, aspect:</strong> Terrain slope and orientation characteristics<br>
                    • <strong>distance_to_coast:</strong> Distance to nearest coastline (km)<br>
                    • <strong>population_density:</strong> Human population density (people/km²)
                </div>
                <div style="margin-top: 10px; padding: 10px; background-color: #e6f3ff; border-radius: 5px; font-size: 10px;">
                    <strong>Interpretation:</strong><br>
                    • Higher importance values indicate stronger influence on species distribution<br>
                    • Variables used in more models are generally more reliable predictors<br>
                    • Top-ranked variables represent key environmental drivers across Indonesian fauna
                </div>
            </div>
        `;
        
        // Create or update the feature summary panel
        var summaryDiv = document.getElementById('featureSummaryPanel');
        if (!summaryDiv) {{
            summaryDiv = document.createElement('div');
            summaryDiv.id = 'featureSummaryPanel';
            summaryDiv.style.cssText = `
                position: fixed; top: 5%; left: 5%; width: 90%; height: 90%;
                background-color: white; border: 2px solid #2E8B57; z-index: 10000;
                border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                display: none;
            `;
            document.body.appendChild(summaryDiv);
        }}
        
        // Add close button and content
        summaryDiv.innerHTML = `
            <div style="position: relative; height: 100%;">
                <button onclick="document.getElementById('featureSummaryPanel').style.display='none';" 
                        style="position: absolute; top: 15px; right: 20px; background: #dc3545; color: white; 
                            border: none; border-radius: 50%; width: 35px; height: 35px; cursor: pointer; 
                            font-weight: bold; z-index: 10001; font-size: 18px;">×</button>
                ${{summaryHtml}}
            </div>
        `;
        
        summaryDiv.style.display = 'block';
    }}

    function changeSpecies(speciesName) {{
        console.log('Changing to species:', speciesName);
        clearAllLayers();
        
        if (speciesName && speciesName !== '') {{
            addMarkersForSpecies(speciesName);
        }} else {{
            // Clear info panel
            var infoDiv = document.getElementById('speciesInfo');
            if (infoDiv) {{
                infoDiv.innerHTML = `
                    <p style="color: #666; font-size: 10px; margin: 0;">
                        Select a species to view its predicted distribution.
                        <br><br>
                        🔴 Red = High habitat suitability (>0.7)<br>
                        🟠 Orange = Medium suitability (0.4-0.7)<br>
                        🟡 Yellow = Low suitability (0.1-0.4)<br>
                        🟢 Green = Observed occurrences<br><br>
                        Click markers for suitability values
                    </p>
                `;
            }}
        }}
    }}

    // Initialize interface after DOM is ready
    setTimeout(function() {{
        try {{
            console.log('Initializing interface...');
            
            // Test map access
            var testMap = findMap();
            if (testMap) {{
                console.log('✅ Map found successfully:', testMap);
                console.log('Map has addLayer method:', typeof testMap.addLayer === 'function');
            }} else {{
                console.error('❌ Map not found during initialization');
                // Try to show available window objects for debugging
                console.log('Available window objects:', Object.keys(window).filter(k => k.includes('map')));
            }}
            
            // Create control panel container
            var controlDiv = document.createElement('div');
            controlDiv.style.cssText = `
                position: fixed; top: 10px; left: 50px; z-index: 9999;
                background-color: white; padding: 8px; border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2); display: flex; align-items: center; gap: 10px;
            `;
            
            // Create species selector
            var select = document.createElement('select');
            select.style.cssText = 'width: 280px; font-size: 11px; padding: 4px;';
            select.innerHTML = '<option value="">🔍 Select a species to view distribution...</option>';
            
            speciesList.forEach(function(species) {{
                var option = document.createElement('option');
                option.value = species;
                option.textContent = species;
                select.appendChild(option);
            }});
            
            select.addEventListener('change', function() {{
                console.log('Species selection changed to:', this.value);
                changeSpecies(this.value);
            }});
            
            // Create feature importance summary button
            var summaryButton = document.createElement('button');
            summaryButton.textContent = '📊 Feature Summary';
            summaryButton.style.cssText = `
                padding: 6px 12px; font-size: 11px; 
                background-color: #2E8B57; color: white; border: none; 
                border-radius: 4px; cursor: pointer; white-space: nowrap;
            `;
            summaryButton.addEventListener('click', showFeatureImportanceSummary);
            
            // Add hover effects
            summaryButton.addEventListener('mouseenter', function() {{
                this.style.backgroundColor = '#1e5c42';
            }});
            summaryButton.addEventListener('mouseleave', function() {{
                this.style.backgroundColor = '#2E8B57';
            }});
            
            controlDiv.appendChild(select);
            controlDiv.appendChild(summaryButton);
            document.body.appendChild(controlDiv);
            
            // Create info panel
            var infoDiv = document.createElement('div');
            infoDiv.id = 'speciesInfo';
            infoDiv.style.cssText = `
                position: fixed; top: 10px; right: 10px; width: 260px;
                background-color: white; border: 1px solid #ccc; z-index: 9999;
                font-size: 11px; padding: 10px; border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2); max-height: 400px;
                overflow-y: auto;
            `;
            infoDiv.innerHTML = `
                <p style="color: #666; font-size: 10px; margin: 0;">
                    Select a species to view its predicted distribution.
                    <br><br>
                    🔴 Red = High habitat suitability (>0.7)<br>
                    🟠 Orange = Medium suitability (0.4-0.7)<br>
                    🟡 Yellow = Low suitability (0.1-0.4)<br>
                    🟢 Green = Observed occurrences<br><br>
                    Click markers for suitability values<br><br>
                    📊 Click "Feature Summary" for overall environmental analysis
                </p>
            `;
            document.body.appendChild(infoDiv);
            
            console.log('Interface initialized successfully');
            console.log('Total species available:', speciesList.length);
            
        }} catch (e) {{
            console.error('Error initializing interface:', e);
        }}
    }}, 2000);  // Increased timeout to ensure map is fully loaded
        """

        # Add JavaScript to map
        m.get_root().html.add_child(folium.Element(f"<script>{js_code}</script>"))
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; bottom: 20px; left: 20px; width: 220px; height: 160px;
                    background-color: white; border: 1px solid grey; z-index: 9999;
                    font-size: 10px; padding: 10px; border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
            <h4 style="margin: 0 0 8px 0; color: #2E8B57; font-size: 12px;">Habitat Suitability</h4>
            
            <div style="margin-bottom: 5px;">
                <span style="display: inline-block; width: 12px; height: 12px; 
                           background-color: red; border-radius: 50%; margin-right: 5px;"></span>
                High (>0.7) - Optimal habitat
            </div>
            
            <div style="margin-bottom: 5px;">
                <span style="display: inline-block; width: 12px; height: 12px; 
                           background-color: orange; border-radius: 50%; margin-right: 5px;"></span>
                Medium (0.4-0.7) - Suitable habitat
            </div>
            
            <div style="margin-bottom: 5px;">
                <span style="display: inline-block; width: 12px; height: 12px; 
                           background-color: yellow; border-radius: 50%; margin-right: 5px;"></span>
                Low (0.1-0.4) - Marginal habitat
            </div>
            
            <div style="margin-bottom: 8px;">
                <span style="display: inline-block; width: 12px; height: 12px; 
                           background-color: lightgreen; border: 1px solid darkgreen; border-radius: 50%; margin-right: 5px;"></span>
                Observed occurrences
            </div>
            
            <div style="font-size: 9px; color: #666; border-top: 1px solid #ccc; padding-top: 5px;">
                • Random Forest predictions<br>
                • Click markers for details<br>
                • Use layer control (top-right)
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m

    def run_fixed_modeling_workflow(self, max_species=10, min_occurrences=20):
        """Run fixed species distribution modeling workflow"""
        start_time = time.time()
        
        print("🔧 FIXED Species Distribution Modeling Workflow")
        print("="*60)
        print("🚀 Robust error handling and debugging included")
        print(f"🎯 Target: {max_species} species with ≥{min_occurrences} occurrences")
        
        # Step 1: Load and prepare environmental data (with fix)
        if not self.load_and_prepare_environmental_data():
            print("❌ Failed to load environmental data")
            return None
        
        # Step 2: Get species list
        species_df = self.get_species_list_optimized(min_occurrences, max_species)
        if len(species_df) == 0:
            print("❌ No suitable species found")
            return None
        
        species_list = species_df['species'].tolist()
        print(f"\n🎯 Selected species for modeling:")
        for i, (_, row) in enumerate(species_df.iterrows(), 1):
            print(f"  {i:2d}. {row['species']:<45} ({row['occurrence_count']:,} records)")
        
        # Step 3: Load all occurrence data at once
        all_occurrences = self.load_all_occurrence_data(species_list)
        if len(all_occurrences) == 0:
            print("❌ Failed to load occurrence data")
            return None
        
        # Step 4: Process each species
        print(f"\n{'='*60}")
        print("TRAINING MODELS")
        print("="*60)
        
        species_models_dict = {}
        
        for i, species_name in enumerate(species_list, 1):
            species_start = time.time()
            print(f"\n[{i:2d}/{len(species_list)}] Processing: {species_name}")
            
            try:
                # Prepare training data
                training_data = self.prepare_training_data_batch(all_occurrences, species_name)
                if training_data is None:
                    print(f"     ❌ Insufficient training data")
                    continue
                
                # Train model
                model_info = self.train_optimized_model(species_name, training_data)
                if model_info is None:
                    print(f"     ❌ Model training failed")
                    continue
                
                # Predict distribution
                predictions = self.predict_distribution_fast(model_info)
                if predictions is None or len(predictions) == 0:
                    print(f"     ❌ Prediction failed")
                    continue
                
                # Store results
                species_models_dict[species_name] = (model_info, predictions)
                
                species_time = time.time() - species_start
                print(f"     ✅ Success! AUC: {model_info['auc_score']:.3f} | "
                      f"Accuracy: {model_info['test_score']:.3f} | "
                      f"Predictions: {len(predictions):,} | "
                      f"Time: {species_time:.1f}s")
                
            except Exception as e:
                print(f"     ❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        if not species_models_dict:
            print("❌ No models were successfully trained")
            return None
        
        # Step 5: Create visualization
        print(f"\n{'='*60}")
        print("CREATING INTERACTIVE MAP")
        print("="*60)
        
        interactive_map = self.create_robust_interactive_map(species_models_dict)
        
        if interactive_map is None:
            print("❌ Failed to create interactive map")
            return None
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_filename = self.results_dir / f"species_distribution_models_fixed_{timestamp}.html"
        interactive_map.save(str(map_filename))
        
        # Save models summary
        summary_data = []
        for species_name, (model_info, predictions) in species_models_dict.items():
            summary_data.append({
                'species': species_name,
                'test_accuracy': model_info['test_score'],
                'auc_score': model_info['auc_score'],
                'n_presences': model_info['n_presences'],
                'n_training_points': model_info['n_training_points'],
                'n_predictions': len(predictions),
                'top_environmental_factor': model_info['feature_importance'].iloc[0]['feature'],
                'top_factor_importance': model_info['feature_importance'].iloc[0]['importance']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = self.results_dir / f"modeling_summary_fixed_{timestamp}.csv"
        summary_df.to_csv(summary_filename, index=False)
        
        # Final summary
        total_time = time.time() - start_time
        avg_time_per_species = total_time / len(species_models_dict)
        
        print(f"\n🎉 FIXED MODELING COMPLETE!")
        print("="*60)
        print(f"✅ Successfully modeled: {len(species_models_dict)}/{len(species_list)} species")
        print(f"⏱️  Total time: {total_time:.1f} seconds ({avg_time_per_species:.1f}s per species)")
        print(f"🗺️  Interactive map: {map_filename}")
        print(f"📊 Summary CSV: {summary_filename}")
        
        print(f"\n📈 Model Performance Overview:")
        avg_auc = summary_df['auc_score'].mean()
        avg_accuracy = summary_df['test_accuracy'].mean()
        best_species = summary_df.loc[summary_df['auc_score'].idxmax(), 'species']
        print(f"   Average AUC: {avg_auc:.3f}")
        print(f"   Average Accuracy: {avg_accuracy:.3f}")
        print(f"   Best performing species: {best_species}")
        
        print(f"\n🔧 Debug Information:")
        print(f"   Environmental points: {len(self.environmental_data):,}")
        print(f"   Environmental variables: {len(self.env_variables)}")
        print(f"   Total occurrence records: {len(all_occurrences):,}")
        
        print(f"\n🔍 Usage Instructions:")
        print(f"   1. Open {map_filename} in your browser")
        print(f"   2. Open browser developer console (F12) to see debug messages")
        print(f"   3. Select species from dropdown menu")
        print(f"   4. View habitat suitability colors:")
        print(f"      • Red = High suitability (>0.7)")
        print(f"      • Orange = Medium suitability (0.4-0.7)")
        print(f"      • Yellow = Low suitability (0.1-0.4)")
        print(f"      • Green = Actual observations")
        print(f"   5. Click markers for detailed suitability values")
        print(f"   6. Right panel shows model performance metrics")
        
        return species_models_dict, str(map_filename)

def main():
    """Main function for fixed species distribution modeling"""
    print("🔧 FIXED Indonesian Species Distribution Modeling")
    print("🚀 Robust Error Handling & Debugging Version")
    print("="*60)
    
    # Initialize modeler
    modeler = FixedSpeciesDistributionModeler()
    
    # Configuration
    max_species = 100  # Reduced for testing
    min_occurrences = 30  # Increased for better models
    
    print(f"⚙️  Configuration:")
    print(f"   • Max species: {max_species}")
    print(f"   • Min occurrences: {min_occurrences}")
    print(f"   • Model: Random Forest (optimized)")
    print(f"   • Fix: Binary blob data conversion + robust JS")
    
    # Check if required data exists
    if not Path("biodiversity_data/indonesia_biodiversity.db").exists():
        print("❌ Database not found! Please run download_records.py first.")
        return
    
    print(f"\n🚀 Starting fixed modeling workflow...")
    
    try:
        result = modeler.run_fixed_modeling_workflow(
            max_species=max_species,
            min_occurrences=min_occurrences
        )
        
        if result:
            models_dict, map_file = result
            print(f"\n✅ SUCCESS! Created {len(models_dict)} species distribution models")
            print(f"\n📁 Files created:")
            print(f"   🗺️  Interactive map: {map_file}")
            print(f"   📊 Results summary: sdm_results/modeling_summary_fixed_*.csv")
            print(f"\n🔧 Debugging Tips:")
            print(f"   • Open browser console (F12) to see debug messages")
            print(f"   • Look for JavaScript errors in console")
            print(f"   • Check that species data is loading correctly")
            print(f"   • Verify map object is accessible")
            print(f"\n🔬 Research Applications:")
            print(f"   • Conservation priority mapping")
            print(f"   • Climate change impact assessment") 
            print(f"   • Habitat suitability analysis")
            print(f"   • Species range predictions")
            
        else:
            print("❌ Modeling workflow failed")
            print("💡 Troubleshooting:")
            print("   • Check that environmental data exists")
            print("   • Verify species occurrence data")
            print("   • Check console for specific error messages")
            
    except Exception as e:
        print(f"❌ Error during modeling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()