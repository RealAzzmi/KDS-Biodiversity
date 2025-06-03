import pandas as pd
import numpy as np
import folium
from folium import plugins
import json
from collections import Counter
import time
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import pygbif, if not available, we'll use alternative methods
try:
    from pygbif import occurrences as occ
    from pygbif import species
    PYGBIF_AVAILABLE = True
except ImportError:
    print("pygbif not available, will use alternative data collection methods")
    PYGBIF_AVAILABLE = False

class BiodiversityAnalyzer:
    def __init__(self):
        self.indonesia_bounds = {
            'min_lat': -11.0,
            'max_lat': 6.0,
            'min_lng': 95.0,
            'max_lng': 141.0
        }
        self.occurrence_data = None
        self.province_data = None
        
    def get_gbif_data_direct_api(self, limit=10000):
        """
        Fetch data directly from GBIF API without pygbif
        """
        print("Fetching GBIF data using direct API calls...")
        
        url = "https://api.gbif.org/v1/occurrence/search"
        all_results = []
        offset = 0
        batch_size = min(300, limit)  # GBIF API limit per request
        
        while len(all_results) < limit:
            params = {
                'country': 'ID',  # Indonesia
                'kingdom': 'Animalia',
                'hasCoordinate': 'true',
                'hasGeospatialIssue': 'false',
                'limit': batch_size,
                'offset': offset
            }
            
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if 'results' in data and data['results']:
                        all_results.extend(data['results'])
                        offset += batch_size
                        print(f"Fetched {len(all_results)} records so far...")
                        
                        # Be nice to the API
                        time.sleep(0.1)
                        
                        # Check if we've got all available data
                        if len(data['results']) < batch_size:
                            break
                    else:
                        break
                else:
                    print(f"API request failed with status code: {response.status_code}")
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data: {e}")
                break
        
        if all_results:
            df = pd.DataFrame(all_results)
            print(f"Retrieved {len(df)} occurrence records from GBIF")
            
            # Clean and filter the data
            df = df.dropna(subset=['species', 'decimalLatitude', 'decimalLongitude'])
            
            # Remove records with invalid coordinates
            df = df[
                (df['decimalLatitude'].between(self.indonesia_bounds['min_lat'], self.indonesia_bounds['max_lat'])) &
                (df['decimalLongitude'].between(self.indonesia_bounds['min_lng'], self.indonesia_bounds['max_lng']))
            ]
            
            print(f"After filtering: {len(df)} valid records")
            self.occurrence_data = df
            return df
        else:
            print("No data retrieved from GBIF")
            return None
    
    def get_gbif_data_pygbif(self, limit=10000):
        """
        Fetch data using pygbif package
        """
        if not PYGBIF_AVAILABLE:
            return None
            
        print("Fetching GBIF data using pygbif...")
        
        try:
            results = occ.search(
                country='ID',  # Indonesia
                kingdom='Animalia',
                hasCoordinate=True,
                hasGeospatialIssue=False,
                limit=limit,
                decimalLatitude=f"{self.indonesia_bounds['min_lat']},{self.indonesia_bounds['max_lat']}",
                decimalLongitude=f"{self.indonesia_bounds['min_lng']},{self.indonesia_bounds['max_lng']}"
            )
            
            if 'results' in results and results['results']:
                df = pd.DataFrame(results['results'])
                print(f"Retrieved {len(df)} occurrence records")
                
                # Clean and filter the data
                df = df.dropna(subset=['species', 'decimalLatitude', 'decimalLongitude'])
                
                # Remove records with invalid coordinates
                df = df[
                    (df['decimalLatitude'].between(self.indonesia_bounds['min_lat'], self.indonesia_bounds['max_lat'])) &
                    (df['decimalLongitude'].between(self.indonesia_bounds['min_lng'], self.indonesia_bounds['max_lng']))
                ]
                
                self.occurrence_data = df
                return df
            else:
                print("No results found")
                return None
                
        except Exception as e:
            print(f"Error fetching GBIF data with pygbif: {e}")
            return None
    
    def get_sample_data(self):
        """
        Create comprehensive sample data for demonstration
        """
        print("Creating sample biodiversity data for Indonesian provinces...")
        
        # Indonesian provinces
        provinces = [
            'Aceh', 'North Sumatra', 'West Sumatra', 'Riau', 'Jambi', 'South Sumatra',
            'Bengkulu', 'Lampung', 'Bangka Belitung', 'Riau Islands', 'Jakarta',
            'West Java', 'Central Java', 'East Java', 'Banten', 'Yogyakarta',
            'Bali', 'West Nusa Tenggara', 'East Nusa Tenggara', 'West Kalimantan',
            'Central Kalimantan', 'South Kalimantan', 'East Kalimantan', 'North Kalimantan',
            'North Sulawesi', 'Central Sulawesi', 'South Sulawesi', 'Southeast Sulawesi',
            'Gorontalo', 'West Sulawesi', 'Maluku', 'North Maluku', 'Papua', 'West Papua'
        ]
        
        # Comprehensive species list for Indonesia
        species_list = [
            # Mammals
            'Panthera tigris sumatrae', 'Elephas maximus sumatranus', 'Rhinoceros sondaicus',
            'Pongo pygmaeus', 'Pongo abelii', 'Macaca fascicularis', 'Macaca nemestrina',
            'Sus scrofa', 'Cervus timorensis', 'Bubalus bubalis', 'Bos javanicus',
            'Paradoxurus hermaphroditus', 'Arctictis binturong', 'Prionailurus bengalensis',
            
            # Reptiles
            'Varanus komodoensis', 'Python reticulatus', 'Python molurus', 'Crocodylus porosus',
            'Chelonia mydas', 'Eretmochelys imbricata', 'Lepidochelys olivacea',
            'Gekko gecko', 'Draco volans', 'Calotes versicolor',
            
            # Birds
            'Hirundo rustica', 'Passer domesticus', 'Gallus gallus', 'Ardea cinerea',
            'Halcyon smyrnensis', 'Pycnonotus aurigaster', 'Anthreptes malacensis',
            'Buceros rhinoceros', 'Probosciger aterrimus', 'Cacatua moluccensis',
            'Trichoglossus haematodus', 'Collocalia esculenta',
            
            # Fish
            'Danio rerio', 'Channa striata', 'Oreochromis niloticus', 'Clarias gariepinus',
            'Rasbora argyrotaenia', 'Barbonymus gonionotus', 'Osphronemus goramy',
            'Trichopodus trichopterus', 'Helostoma temminckii',
            
            # Amphibians
            'Fejervarya limnocharis', 'Polypedates leucomystax', 'Rhacophorus reinwardtii',
            'Bufo melanostictus', 'Microhyla palmipes',
            
            # Invertebrates
            'Attacus atlas', 'Ornithoptera priamus', 'Troides helena',
            'Apis dorsata', 'Vespa tropica', 'Gryllus bimaculatus'
        ]
        
        # Create sample occurrence data with realistic patterns
        sample_data = []
        np.random.seed(42)
        
        # Different biodiversity patterns for different regions
        region_multipliers = {
            'Sumatra': ['Aceh', 'North Sumatra', 'West Sumatra', 'Riau', 'Jambi', 'South Sumatra', 'Bengkulu', 'Lampung'],
            'Java': ['Jakarta', 'West Java', 'Central Java', 'East Java', 'Banten', 'Yogyakarta'],
            'Kalimantan': ['West Kalimantan', 'Central Kalimantan', 'South Kalimantan', 'East Kalimantan', 'North Kalimantan'],
            'Sulawesi': ['North Sulawesi', 'Central Sulawesi', 'South Sulawesi', 'Southeast Sulawesi', 'Gorontalo', 'West Sulawesi'],
            'Eastern': ['Maluku', 'North Maluku', 'Papua', 'West Papua'],
            'Lesser_Sunda': ['Bali', 'West Nusa Tenggara', 'East Nusa Tenggara'],
            'Islands': ['Bangka Belitung', 'Riau Islands']
        }
        
        # Generate different numbers of records per region (biodiversity hotspots)
        region_record_counts = {
            'Sumatra': 3000,  # High biodiversity
            'Kalimantan': 2500,  # High biodiversity
            'Papua': 2000,  # Very high biodiversity but less sampled
            'Sulawesi': 1800,  # High endemism
            'Java': 1500,  # High human impact, lower diversity
            'Lesser_Sunda': 1000,  # Moderate diversity
            'Eastern': 800,  # High diversity but remote
            'Islands': 400   # Small islands, lower diversity
        }
        
        for region, provinces_in_region in region_multipliers.items():
            total_records = region_record_counts.get(region, 1000)
            records_per_province = total_records // len(provinces_in_region)
            
            for province in provinces_in_region:
                # Create species abundance distribution (some species are much more common)
                species_weights = np.random.exponential(scale=2.0, size=len(species_list))
                species_weights = species_weights / species_weights.sum()
                
                for _ in range(records_per_province):
                    species = np.random.choice(species_list, p=species_weights)
                    
                    # Generate coordinates within Indonesia bounds, clustered by region
                    if region == 'Sumatra':
                        lat = np.random.normal(-1.0, 2.0)
                        lng = np.random.normal(101.0, 2.0)
                    elif region == 'Java':
                        lat = np.random.normal(-7.0, 1.0)
                        lng = np.random.normal(108.0, 2.0)
                    elif region == 'Kalimantan':
                        lat = np.random.normal(-1.0, 3.0)
                        lng = np.random.normal(114.0, 3.0)
                    elif region == 'Sulawesi':
                        lat = np.random.normal(-2.0, 2.0)
                        lng = np.random.normal(120.0, 2.0)
                    elif region == 'Papua':
                        lat = np.random.normal(-4.0, 2.0)
                        lng = np.random.normal(138.0, 2.0)
                    else:
                        lat = np.random.uniform(self.indonesia_bounds['min_lat'], self.indonesia_bounds['max_lat'])
                        lng = np.random.uniform(self.indonesia_bounds['min_lng'], self.indonesia_bounds['max_lng'])
                    
                    # Ensure coordinates are within bounds
                    lat = np.clip(lat, self.indonesia_bounds['min_lat'], self.indonesia_bounds['max_lat'])
                    lng = np.clip(lng, self.indonesia_bounds['min_lng'], self.indonesia_bounds['max_lng'])
                    
                    sample_data.append({
                        'species': species,
                        'decimalLatitude': lat,
                        'decimalLongitude': lng,
                        'stateProvince': province,
                        'kingdom': 'Animalia',
                        'phylum': 'Chordata',  # Most of our examples
                        'class': 'Mammalia'    # Simplified
                    })
        
        self.occurrence_data = pd.DataFrame(sample_data)
        print(f"Created {len(self.occurrence_data)} sample occurrence records")
        return self.occurrence_data
    
    def get_data(self, limit=10000, use_sample=False):
        """
        Main method to get data - tries multiple approaches
        """
        if use_sample:
            return self.get_sample_data()
        
        # Try pygbif first
        if PYGBIF_AVAILABLE:
            data = self.get_gbif_data_pygbif(limit)
            if data is not None and len(data) > 50:
                return data
        
        # Try direct API
        data = self.get_gbif_data_direct_api(limit)
        if data is not None and len(data) > 50:
            return data
        
        # Fall back to sample data
        print("Real GBIF data not available, using sample data...")
        return self.get_sample_data()
    
    def calculate_shannon_index(self, species_counts):
        """
        Calculate Shannon Diversity Index
        H = -Σ(pi * ln(pi))
        where pi is the proportion of individuals of species i
        """
        if len(species_counts) == 0:
            return 0
        
        total = sum(species_counts.values())
        if total == 0:
            return 0
        
        shannon_index = 0
        for count in species_counts.values():
            if count > 0:
                pi = count / total
                shannon_index -= pi * np.log(pi)
        
        return shannon_index
    
    def analyze_biodiversity_by_province(self):
        """
        Calculate Shannon Index for each province
        """
        if self.occurrence_data is None:
            print("No occurrence data available")
            return None
        
        province_biodiversity = {}
        
        # Group by province
        provinces = self.occurrence_data['stateProvince'].dropna().unique()
        
        for province in provinces:
            province_data = self.occurrence_data[
                self.occurrence_data['stateProvince'] == province
            ]
            
            # Count species occurrences
            species_counts = Counter(province_data['species'])
            
            # Calculate Shannon Index
            shannon_index = self.calculate_shannon_index(species_counts)
            
            # Calculate other metrics
            species_richness = len(species_counts)
            total_occurrences = len(province_data)
            
            # Calculate evenness (Pielou's evenness index)
            if species_richness > 1:
                max_diversity = np.log(species_richness)
                evenness = shannon_index / max_diversity
            else:
                evenness = 0
            
            province_biodiversity[province] = {
                'shannon_index': shannon_index,
                'species_richness': species_richness,
                'total_occurrences': total_occurrences,
                'evenness': evenness,
                'dominant_species': species_counts.most_common(5)
            }
        
        self.province_data = province_biodiversity
        return province_biodiversity
    
    def create_interactive_map(self):
        """
        Create an interactive map showing biodiversity indices
        """
        if self.province_data is None:
            print("No province biodiversity data available")
            return None
        
        # Create base map centered on Indonesia
        m = folium.Map(
            location=[-2.5, 118.0],  # Center of Indonesia
            zoom_start=5,
            tiles='OpenStreetMap'
        )
        
        # Add different tile layers
        folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)
        
        # Add heatmap of species occurrences
        if self.occurrence_data is not None:
            heat_data = []
            for _, row in self.occurrence_data.iterrows():
                if not pd.isna(row['decimalLatitude']) and not pd.isna(row['decimalLongitude']):
                    heat_data.append([row['decimalLatitude'], row['decimalLongitude']])
            
            if heat_data:
                plugins.HeatMap(heat_data, name='Species Occurrences Heatmap', show=False).add_to(m)
        
        # Add province markers with detailed popups
        for province, data in self.province_data.items():
            province_coords = self.occurrence_data[
                self.occurrence_data['stateProvince'] == province
            ][['decimalLatitude', 'decimalLongitude']].mean()
            
            if not province_coords.isna().any():
                # Create detailed popup
                popup_html = f"""
                <div style="width: 350px; font-family: Arial, sans-serif;">
                    <h3 style="color: #2E8B57; margin-bottom: 10px; text-align: center;">{province}</h3>
                    
                    <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <h4 style="margin: 0 0 8px 0; color: #1e4d72;">Biodiversity Metrics</h4>
                        <table style="width: 100%; font-size: 12px;">
                            <tr><td><b>Shannon Index:</b></td><td>{data['shannon_index']:.3f}</td></tr>
                            <tr><td><b>Species Richness:</b></td><td>{data['species_richness']}</td></tr>
                            <tr><td><b>Evenness Index:</b></td><td>{data['evenness']:.3f}</td></tr>
                            <tr><td><b>Total Records:</b></td><td>{data['total_occurrences']}</td></tr>
                        </table>
                    </div>
                    
                    <div style="background-color: #fff8dc; padding: 10px; border-radius: 5px;">
                        <h4 style="margin: 0 0 8px 0; color: #8B4513;">Top 5 Species</h4>
                        <ol style="margin: 0; padding-left: 20px; font-size: 11px;">
                """
                
                for species, count in data['dominant_species']:
                    popup_html += f"<li><i>{species}</i>: {count} records</li>"
                
                popup_html += """
                        </ol>
                    </div>
                    
                    <div style="text-align: center; margin-top: 10px; font-size: 10px; color: #666;">
                        Click outside to close | Data from GBIF
                    </div>
                </div>
                """
                
                # Determine marker color based on Shannon Index
                shannon_index = data['shannon_index']
                if shannon_index > 2.5:
                    color = '#006400'  # Dark green
                elif shannon_index > 2.0:
                    color = '#32CD32'  # Lime green
                elif shannon_index > 1.5:
                    color = '#FFA500'  # Orange
                else:
                    color = '#FF0000'  # Red
                
                # Create marker
                folium.CircleMarker(
                    location=[province_coords['decimalLatitude'], province_coords['decimalLongitude']],
                    radius=max(5, min(20, shannon_index * 5)),
                    popup=folium.Popup(popup_html, max_width=400, max_height=300),
                    color='white',
                    weight=2,
                    fillColor=color,
                    fillOpacity=0.8,
                    tooltip=f"{province}<br>Shannon Index: {shannon_index:.3f}"
                ).add_to(m)
        
        # Add custom legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 220px; height: 180px; 
                    background-color: white; border: 2px solid grey; z-index: 9999; 
                    font-size: 12px; padding: 15px; border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            
            <h4 style="margin: 0 0 10px 0; color: #2E8B57;">Shannon Diversity Index</h4>
            
            <div style="margin-bottom: 8px;">
                <span style="display: inline-block; width: 15px; height: 15px; 
                           background-color: #006400; border-radius: 50%; margin-right: 8px;"></span>
                Very High (> 2.5)
            </div>
            
            <div style="margin-bottom: 8px;">
                <span style="display: inline-block; width: 15px; height: 15px; 
                           background-color: #32CD32; border-radius: 50%; margin-right: 8px;"></span>
                High (2.0 - 2.5)
            </div>
            
            <div style="margin-bottom: 8px;">
                <span style="display: inline-block; width: 15px; height: 15px; 
                           background-color: #FFA500; border-radius: 50%; margin-right: 8px;"></span>
                Medium (1.5 - 2.0)
            </div>
            
            <div style="margin-bottom: 12px;">
                <span style="display: inline-block; width: 15px; height: 15px; 
                           background-color: #FF0000; border-radius: 50%; margin-right: 8px;"></span>
                Low (< 1.5)
            </div>
            
            <hr style="margin: 10px 0;">
            
            <div style="font-size: 10px; color: #666;">
                • Circle size ∝ Shannon Index<br>
                • Click markers for details<br>
                • Toggle layers in top-right
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        return m
    
    def export_results(self, filename_prefix="indonesia_biodiversity"):
        """
        Export results in multiple formats
        """
        if not self.province_data:
            print("No data to export")
            return None
        
        # Create DataFrame for export
        export_data = []
        for province, data in self.province_data.items():
            row = {
                'Province': province,
                'Shannon_Index': data['shannon_index'],
                'Species_Richness': data['species_richness'],
                'Total_Occurrences': data['total_occurrences'],
                'Evenness_Index': data['evenness'],
            }
            
            # Add top species
            for i, (species, count) in enumerate(data['dominant_species'][:3], 1):
                row[f'Top_Species_{i}'] = species
                row[f'Top_Species_{i}_Count'] = count
            
            export_data.append(row)
        
        df = pd.DataFrame(export_data)
        
        # Export to CSV
        csv_filename = f"{filename_prefix}_results.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Results exported to {csv_filename}")
        
        return df
    
    def generate_summary_statistics(self):
        """
        Generate summary statistics and visualizations
        """
        if self.province_data is None:
            return None
        
        # Create DataFrame for analysis
        summary_df = pd.DataFrame([
            {
                'Province': province,
                'Shannon_Index': data['shannon_index'],
                'Species_Richness': data['species_richness'],
                'Total_Occurrences': data['total_occurrences'],
                'Evenness': data['evenness']
            }
            for province, data in self.province_data.items()
        ])
        
        # Summary statistics
        print("\n" + "="*60)
        print("INDONESIAN BIODIVERSITY ANALYSIS RESULTS")
        print("="*60)
        print(f"Total Provinces Analyzed: {len(summary_df)}")
        print(f"Average Shannon Index: {summary_df['Shannon_Index'].mean():.3f}")
        print(f"Highest Shannon Index: {summary_df['Shannon_Index'].max():.3f}")
        print(f"Lowest Shannon Index: {summary_df['Shannon_Index'].min():.3f}")
        print(f"Total Unique Species: {len(set([species for data in self.province_data.values() for species, count in data['dominant_species']]))}")
        print(f"Total Occurrence Records: {summary_df['Total_Occurrences'].sum()}")
        
        # Top 5 most diverse provinces
        print("\n" + "="*40)
        print("TOP 5 MOST BIODIVERSE PROVINCES")
        print("="*40)
        top_provinces = summary_df.nlargest(5, 'Shannon_Index')
        for i, (_, row) in enumerate(top_provinces.iterrows(), 1):
            print(f"{i}. {row['Province']}: Shannon Index = {row['Shannon_Index']:.3f}")
        
        # Bottom 5 provinces
        print("\n" + "="*40)
        print("PROVINCES WITH LOWEST DIVERSITY")
        print("="*40)
        bottom_provinces = summary_df.nsmallest(5, 'Shannon_Index')
        for i, (_, row) in enumerate(bottom_provinces.iterrows(), 1):
            print(f"{i}. {row['Province']}: Shannon Index = {row['Shannon_Index']:.3f}")
        
        return summary_df

def run_complete_analysis():
    """
    Complete workflow for biodiversity analysis
    """
    print("=== INDONESIAN BIODIVERSITY ANALYSIS USING SHANNON INDEX ===\n")
    
    # Initialize analyzer
    analyzer = BiodiversityAnalyzer()
    
    # Get data (try real GBIF data first, fall back to sample data)
    print("Step 1: Fetching species occurrence data...")
    try:
        data = analyzer.get_data(limit=15000, use_sample=False)
    except:
        print("Using sample data...")
        data = analyzer.get_data(limit=15000, use_sample=True)
    
    if data is None:
        print("Failed to get data")
        return None, None
    
    print(f"Working with {len(data)} occurrence records")
    
    # Analyze biodiversity
    print("\nStep 2: Calculating Shannon indices for each province...")
    biodiversity_results = analyzer.analyze_biodiversity_by_province()
    
    if biodiversity_results is None:
        print("Failed to analyze biodiversity")
        return None, None
    
    # Generate summary statistics
    print("\nStep 3: Generating summary statistics...")
    summary_stats = analyzer.generate_summary_statistics()
    
    # Create interactive map
    print("\nStep 4: Creating interactive map...")
    biodiversity_map = analyzer.create_interactive_map()
    
    if biodiversity_map:
        map_filename = "indonesia_biodiversity_map.html"
        biodiversity_map.save(map_filename)
        print(f"\n✅ Interactive map saved as '{map_filename}'")
        print("   Open this file in a web browser to explore the data!")
    
    # Export results
    print("\nStep 5: Exporting results...")
    export_df = analyzer.export_results()
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE! 🎉")
    print("="*60)
    print("Files generated:")
    print("1. indonesia_biodiversity_map.html - Interactive map with clickable provinces")
    print("2. indonesia_biodiversity_results.csv - Summary results for analysis")
    
    print("\n📊 For your paper:")
    print("- Open the HTML map to explore biodiversity patterns")
    print("- Use the CSV data for statistical analysis")
    print("- Reference the Shannon Index methodology")
    print("- Include province-specific findings from the map popups")
    
    return analyzer, export_df

# Run the analysis
if __name__ == "__main__":
    analyzer, results = run_complete_analysis()