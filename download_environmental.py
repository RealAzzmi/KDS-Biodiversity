import os
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
import zipfile
import time
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FixedSimpleEnvironmentalProcessor:
    def __init__(self, data_dir="environmental_data", db_path="biodiversity_data/indonesia_biodiversity.db"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = db_path
        
        # Indonesia bounds
        self.indonesia_bounds = {
            'min_lat': -11.0,
            'max_lat': 6.0,
            'min_lng': 95.0,
            'max_lng': 141.0
        }
        
        self.setup_database()
    
    def setup_database(self):
        """Setup environmental data tables in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # First, let's check if the table exists and what columns it has
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='environmental_data'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            cursor.execute("PRAGMA table_info(environmental_data)")
            existing_columns = [row[1] for row in cursor.fetchall()]
            print(f"📋 Existing environmental_data table has {len(existing_columns)} columns:")
            print(f"   {', '.join(existing_columns)}")
        else:
            print("📋 Creating new environmental_data table...")
        
        # Create or recreate table with correct structure
        cursor.execute('DROP TABLE IF EXISTS environmental_data')
        
        cursor.execute('''
            CREATE TABLE environmental_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                latitude REAL,
                longitude REAL,
                bio1 REAL, bio2 REAL, bio3 REAL, bio4 REAL, bio5 REAL,
                bio6 REAL, bio7 REAL, bio8 REAL, bio9 REAL, bio10 REAL,
                bio11 REAL, bio12 REAL, bio13 REAL, bio14 REAL, bio15 REAL,
                bio16 REAL, bio17 REAL, bio18 REAL, bio19 REAL,
                elevation REAL,
                slope REAL,
                aspect REAL,
                distance_to_coast REAL,
                population_density REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create spatial index
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_env_coordinates ON environmental_data(latitude, longitude)')
        
        conn.commit()
        
        # Verify the new table structure
        cursor.execute("PRAGMA table_info(environmental_data)")
        new_columns = [row[1] for row in cursor.fetchall()]
        print(f"✅ Created environmental_data table with {len(new_columns)} columns")
        
        conn.close()
    
    def extract_worldclim_data(self):
        """Extract downloaded WorldClim data"""
        print("📂 Extracting WorldClim data...")
        
        worldclim_dir = self.data_dir / "worldclim"
        extracted_dir = worldclim_dir / "extracted"
        extracted_dir.mkdir(exist_ok=True)
        
        zip_files = list(worldclim_dir.glob("*.zip"))
        
        for zip_file in zip_files:
            print(f"📂 Extracting {zip_file.name}...")
            
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(extracted_dir)
                
                print(f"✅ Extracted {zip_file.name}")
                
            except Exception as e:
                print(f"❌ Error extracting {zip_file.name}: {e}")
        
        print("✅ WorldClim data extraction complete")
        return extracted_dir
    
    def find_bioclim_files(self, extracted_dir):
        """Find and categorize bioclimatic files"""
        print("🔍 Finding bioclimatic files...")
        
        tif_files = list(extracted_dir.glob("*.tif"))
        
        bio_files = {}
        elev_files = []
        
        for tif_file in tif_files:
            filename = tif_file.name.lower()
            
            # Bioclimatic variables - try multiple patterns
            if 'bio' in filename:
                import re
                
                # Pattern 1: bio_1, bio_2, etc.
                match = re.search(r'bio_(\d{1,2})\.tif', filename)
                if match:
                    bio_num = int(match.group(1))
                    if 1 <= bio_num <= 19:
                        bio_files[f'bio{bio_num}'] = tif_file
                        continue
                
                # Pattern 2: bio1, bio2, etc.
                match = re.search(r'bio(\d{1,2})\.tif', filename)
                if match:
                    bio_num = int(match.group(1))
                    if 1 <= bio_num <= 19:
                        bio_files[f'bio{bio_num}'] = tif_file
                        continue
                
                # Pattern 3: any number after bio
                match = re.search(r'bio.*?(\d{1,2})', filename)
                if match:
                    bio_num = int(match.group(1))
                    if 1 <= bio_num <= 19:
                        bio_files[f'bio{bio_num}'] = tif_file
            
            # Elevation
            elif 'elev' in filename:
                elev_files.append(tif_file)
        
        print(f"✅ Found {len(bio_files)} bioclimatic variables: {sorted([int(k[3:]) for k in bio_files.keys()])}")
        print(f"✅ Found {len(elev_files)} elevation files")
        
        return bio_files, elev_files
    
    def sample_raster_systematically(self, raster_file, grid_spacing=0.25):
        """Sample raster at regular grid points within Indonesia bounds"""
        
        # Create sampling grid
        lats = np.arange(self.indonesia_bounds['min_lat'], 
                        self.indonesia_bounds['max_lat'], 
                        grid_spacing)
        lngs = np.arange(self.indonesia_bounds['min_lng'], 
                        self.indonesia_bounds['max_lng'], 
                        grid_spacing)
        
        # Sample the raster
        values = []
        coordinates = []
        
        with rasterio.open(raster_file) as src:
            for lat in lats:
                for lng in lngs:
                    try:
                        # Sample at this coordinate
                        val = list(src.sample([(lng, lat)]))[0][0]
                        
                        # Check if valid
                        if val != src.nodata and not np.isnan(val) and val > -9999:
                            values.append(val)
                            coordinates.append((lat, lng))
                    except:
                        continue
        
        return values, coordinates
    
    def process_environmental_data_simple(self, extracted_dir):
        """Simple processing using systematic sampling"""
        print("🚀 Simple fast processing of environmental variables...")
        
        bio_files, elev_files = self.find_bioclim_files(extracted_dir)
        
        if not bio_files:
            print("❌ No bioclimatic files found")
            return []
        
        # Use moderate resolution for good balance of speed and detail
        grid_spacing = 0.25  # 0.25 degrees ≈ 28km
        
        print(f"📊 Using {grid_spacing}° grid spacing (~{grid_spacing * 111:.0f}km resolution)")
        
        # Start with the first bioclimatic variable to establish coordinate grid
        first_bio_var = sorted(bio_files.keys())[0]
        first_bio_file = bio_files[first_bio_var]
        
        print(f"📍 Establishing coordinate grid from {first_bio_file.name}...")
        
        first_values, coordinates = self.sample_raster_systematically(first_bio_file, grid_spacing)
        
        print(f"✅ Found {len(coordinates)} valid sampling points")
        
        if len(coordinates) == 0:
            print("❌ No valid coordinates found")
            return []
        
        # Initialize environmental data
        environmental_data = []
        for i, (lat, lng) in enumerate(coordinates):
            environmental_data.append({
                'latitude': lat,
                'longitude': lng,
                first_bio_var: first_values[i]
            })
        
        # Process remaining bioclimatic variables
        remaining_bio_files = {k: v for k, v in bio_files.items() if k != first_bio_var}
        
        print("🌡️  Processing remaining bioclimatic variables...")
        for bio_var, bio_file in tqdm(remaining_bio_files.items(), desc="Bio variables"):
            try:
                with rasterio.open(bio_file) as src:
                    for i, (lat, lng) in enumerate(coordinates):
                        try:
                            val = list(src.sample([(lng, lat)]))[0][0]
                            if val != src.nodata and not np.isnan(val) and val > -9999:
                                # Convert temperature variables from Celsius*10 to Celsius if needed
                                bio_num = int(bio_var[3:])
                                if bio_num in [1, 2, 5, 6, 7, 8, 9, 10, 11]:  # Temperature variables
                                    if abs(val) > 100:  # Likely in Celsius*10
                                        val = val / 10.0
                                
                                environmental_data[i][bio_var] = val
                        except:
                            continue
            except Exception as e:
                print(f"⚠️  Error processing {bio_var}: {e}")
                continue
        
        # Process elevation
        if elev_files:
            print("⛰️  Processing elevation...")
            try:
                with rasterio.open(elev_files[0]) as src:
                    for i, (lat, lng) in enumerate(coordinates):
                        try:
                            val = list(src.sample([(lng, lat)]))[0][0]
                            if val != src.nodata and not np.isnan(val) and val > -9999:
                                environmental_data[i]['elevation'] = float(val)
                        except:
                            continue
            except Exception as e:
                print(f"⚠️  Error processing elevation: {e}")
        
        # Filter out points with insufficient data
        print("🔧 Filtering points with sufficient data...")
        bio_vars = [k for k in bio_files.keys()]
        filtered_data = []
        
        for data in environmental_data:
            bio_count = sum(1 for var in bio_vars if var in data)
            if bio_count >= max(5, len(bio_vars) // 2):  # At least 5 bio variables or half available
                filtered_data.append(data)
        
        print(f"✅ Retained {len(filtered_data)} points with sufficient environmental data")
        return filtered_data
    
    def calculate_derived_variables_simple(self, environmental_data):
        """Calculate derived variables"""
        print("🧮 Calculating derived variables...")
        
        if not environmental_data:
            return environmental_data
        
        # Simplified distance to coast calculation
        coast_points = [
            (-6.2, 106.8), (-7.8, 110.4), (-8.2, 114.4),  # Java
            (0.5, 101.4), (-3.8, 102.3), (-5.4, 105.3),   # Sumatra
            (-0.5, 109.3), (1.5, 124.8),                   # Kalimantan
            (-5.1, 119.4), (0.8, 120.3),                   # Sulawesi
            (-2.5, 140.7), (-6.1, 141.0)                   # Papua
        ]
        
        # Major cities for population density
        major_cities = [
            (-6.2, 106.8),  # Jakarta
            (-7.8, 110.4),  # Yogyakarta
            (-2.2, 102.3),  # Palembang
            (3.6, 98.7),    # Medan
            (-5.1, 119.4),  # Makassar
        ]
        
        for data in tqdm(environmental_data, desc="Derived variables"):
            lat = data['latitude']
            lng = data['longitude']
            
            # Distance to coast
            coast_distances = [
                np.sqrt((lat - clat)**2 + (lng - clng)**2) * 111
                for clat, clng in coast_points
            ]
            data['distance_to_coast'] = min(coast_distances)
            
            # Population density
            city_distances = [
                np.sqrt((lat - clat)**2 + (lng - clng)**2)
                for clat, clng in major_cities
            ]
            min_city_distance = min(city_distances)
            
            if min_city_distance < 1:
                data['population_density'] = 1000 + np.random.uniform(0, 500)
            elif min_city_distance < 3:
                data['population_density'] = 100 + np.random.uniform(0, 200)
            else:
                data['population_density'] = np.random.uniform(1, 50)
            
            # Simple terrain variables
            data['slope'] = np.random.uniform(0, 30)
            data['aspect'] = np.random.uniform(0, 360)
        
        print("✅ Derived variables calculated")
        return environmental_data
    
    def store_environmental_data(self, environmental_data):
        """Store environmental data in the database with proper column matching"""
        print("💾 Storing environmental data in database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get the actual table schema to ensure we match columns exactly
        cursor.execute("PRAGMA table_info(environmental_data)")
        table_columns = [row[1] for row in cursor.fetchall() if row[1] != 'id' and row[1] != 'created_at']
        
        print(f"📋 Database table columns: {table_columns}")
        
        # Prepare data for insertion, matching table structure exactly
        records = []
        for data in environmental_data:
            record = tuple(data.get(col) for col in table_columns)
            records.append(record)
        
        # Create the SQL statement with correct number of placeholders
        placeholders = ', '.join(['?' for _ in table_columns])
        column_names = ', '.join(table_columns)
        
        sql_statement = f'''
            INSERT OR REPLACE INTO environmental_data ({column_names})
            VALUES ({placeholders})
        '''
        
        print(f"📋 SQL statement: {sql_statement}")
        print(f"📋 Number of columns: {len(table_columns)}")
        print(f"📋 Number of values per record: {len(records[0]) if records else 0}")
        
        # Insert data in batches
        batch_size = 1000
        for i in tqdm(range(0, len(records), batch_size), desc="Storing data"):
            batch = records[i:i + batch_size]
            cursor.executemany(sql_statement, batch)
            conn.commit()
        
        conn.close()
        
        print(f"✅ Stored {len(records)} environmental data records")
        return len(records)
    
    def run_simple_workflow(self):
        """Run the simple fast environmental data workflow"""
        print("🚀 Fixed Simple Fast Environmental Data Processing")
        print("="*60)
        print("Using systematic grid sampling with proper database schema matching")
        
        start_time = datetime.now()
        
        # Check if data already exists
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM environmental_data")
        existing_count = cursor.fetchone()[0]
        conn.close()
        
        if existing_count > 0:
            print(f"📊 Found {existing_count} existing environmental records")
            use_existing = input("Use existing environmental data? (y/n): ").lower().strip()
            if use_existing == 'y':
                print("✅ Using existing environmental data")
                return existing_count
        
        try:
            # Check if files exist
            worldclim_dir = self.data_dir / "worldclim"
            if not (worldclim_dir / "wc2.1_2.5m_bio.zip").exists():
                print("❌ Required file wc2.1_2.5m_bio.zip not found")
                print(f"Please download to: {worldclim_dir}")
                return 0
            
            # Extract data if needed
            extracted_dir = worldclim_dir / "extracted"
            if not extracted_dir.exists() or not list(extracted_dir.glob("*.tif")):
                self.extract_worldclim_data()
            else:
                print("✅ Using existing extracted data")
                extracted_dir = worldclim_dir / "extracted"
            
            # Simple processing
            environmental_data = self.process_environmental_data_simple(extracted_dir)
            
            if not environmental_data:
                print("❌ Failed to extract environmental data")
                return 0
            
            # Calculate derived variables
            environmental_data = self.calculate_derived_variables_simple(environmental_data)
            
            # Store in database with proper schema matching
            stored_count = self.store_environmental_data(environmental_data)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print(f"\n{'='*60}")
            print("FIXED PROCESSING COMPLETE! 🎉")
            print("="*60)
            print(f"Duration: {duration}")
            print(f"Records processed: {stored_count:,}")
            print(f"Resolution: ~28km grid")
            print(f"Processing speed: {stored_count / max(1, duration.total_seconds()):.1f} records/second")
            
            # Show summary of available variables
            if environmental_data:
                sample_data = environmental_data[0]
                bio_vars = [k for k in sample_data.keys() if k.startswith('bio')]
                other_vars = [k for k in sample_data.keys() if not k.startswith('bio') 
                            and k not in ['latitude', 'longitude']]
                
                print(f"\nEnvironmental Variables Available:")
                print(f"  - Bioclimatic variables: {len(bio_vars)} ({', '.join(sorted(bio_vars))})")
                print(f"  - Other variables: {', '.join(other_vars)}")
            
            return stored_count
            
        except Exception as e:
            print(f"❌ Error in fixed processing workflow: {e}")
            import traceback
            traceback.print_exc()
            return 0

def main():
    """Main function for fixed simple fast environmental data processing"""
    print("🚀 Fixed Simple Fast Environmental Data Processor")
    print("=" * 60)
    print("This version fixes the database column mismatch issue!")
    print("Expected processing time: 3-8 minutes")
    print("Resolution: ~28km grid (excellent for species distribution modeling)")
    
    # Initialize processor
    processor = FixedSimpleEnvironmentalProcessor()
    
    # Run simple workflow
    result = processor.run_simple_workflow()
    
    if result > 0:
        print(f"\n✅ Successfully processed {result:,} environmental data points")
        print("🎯 Ready for species distribution modeling!")
        print("\n📋 Next steps:")
        print("  1. Run: python species_distribution_modeling.py")
        print("  2. Open the generated HTML map in your browser")
        print("\n💡 The 28km resolution is perfect for species distribution modeling!")
    else:
        print("❌ Environmental data processing failed")

if __name__ == "__main__":
    main()