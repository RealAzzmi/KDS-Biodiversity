import pandas as pd
import numpy as np
import folium
from folium import plugins
import json
import pickle
import os
from collections import Counter
import time
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sqlite3
from pathlib import Path

# Try to import pygbif, if not available, we'll use alternative methods
try:
    from pygbif import occurrences as occ
    from pygbif import species
    PYGBIF_AVAILABLE = True
except ImportError:
    print("pygbif not available, will use alternative data collection methods")
    PYGBIF_AVAILABLE = False

class ComprehensiveBiodiversityAnalyzer:
    def __init__(self, data_dir="biodiversity_data"):
        self.indonesia_bounds = {
            'min_lat': -11.0,
            'max_lat': 6.0,
            'min_lng': 95.0,
            'max_lng': 141.0
        }
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Database setup for efficient storage and querying
        self.db_path = self.data_dir / "indonesia_biodiversity.db"
        self.setup_database()
        
        self.occurrence_data = None
        self.province_data = None
        self.crawl_metadata = {
            'total_records': 0,
            'last_offset': 0,
            'start_time': None,
            'end_time': None,
            'crawl_sessions': []
        }
        
        # Load existing metadata if available
        self.load_crawl_metadata()
        
    def setup_database(self):
        """Setup SQLite database for efficient data storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create main occurrences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS occurrences (
                gbifID INTEGER PRIMARY KEY,
                species TEXT,
                scientificName TEXT,
                kingdom TEXT,
                phylum TEXT,
                class TEXT,
                order_name TEXT,
                family TEXT,
                genus TEXT,
                decimalLatitude REAL,
                decimalLongitude REAL,
                country TEXT,
                stateProvince TEXT,
                locality TEXT,
                year INTEGER,
                month INTEGER,
                day INTEGER,
                basisOfRecord TEXT,
                institutionCode TEXT,
                collectionCode TEXT,
                catalogNumber TEXT,
                recordedBy TEXT,
                eventDate TEXT,
                coordinateUncertaintyInMeters REAL,
                elevation REAL,
                depth REAL,
                crawl_session TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_species ON occurrences(species)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_province ON occurrences(stateProvince)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_coordinates ON occurrences(decimalLatitude, decimalLongitude)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_gbifid ON occurrences(gbifID)')
        
        # Create metadata table for tracking crawl progress
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crawl_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                total_records INTEGER,
                last_offset INTEGER,
                status TEXT,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def load_crawl_metadata(self):
        """Load existing crawl metadata"""
        metadata_file = self.data_dir / "crawl_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.crawl_metadata = json.load(f)
                
    def save_crawl_metadata(self):
        """Save crawl metadata"""
        metadata_file = self.data_dir / "crawl_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.crawl_metadata, f, indent=2, default=str)
            
    def get_existing_record_count(self):
        """Get count of existing records in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM occurrences")
        count = cursor.fetchone()[0]
        conn.close()
        return count
        
    def get_existing_gbif_ids(self, limit=100000):
        """Get existing GBIF IDs to avoid duplicates"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT gbifID FROM occurrences LIMIT ?", (limit,))
        existing_ids = set(row[0] for row in cursor.fetchall())
        conn.close()
        return existing_ids
        
    def comprehensive_gbif_crawl(self, max_records=None, resume=True):
        """
        Comprehensive crawl of all Animalia data from Indonesia
        """
        session_id = f"crawl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Starting comprehensive GBIF crawl - Session: {session_id}")
        
        # Check existing data
        existing_count = self.get_existing_record_count()
        print(f"Existing records in database: {existing_count:,}")
        
        if existing_count > 0 and resume:
            print("Resume mode: Will skip existing records and continue crawling")
            existing_ids = self.get_existing_gbif_ids()
        else:
            existing_ids = set()
            
        url = "https://api.gbif.org/v1/occurrence/search"
        all_results = []
        offset = self.crawl_metadata.get('last_offset', 0) if resume else 0
        batch_size = 300  # GBIF API limit per request
        session_records = 0
        total_api_calls = 0
        
        # Start crawl session
        start_time = datetime.now()
        self.crawl_metadata['start_time'] = start_time
        self.crawl_metadata['crawl_sessions'].append({
            'session_id': session_id,
            'start_time': start_time,
            'status': 'running'
        })
        
        # Record session in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO crawl_metadata 
            (session_id, start_time, status, last_offset) 
            VALUES (?, ?, ?, ?)
        ''', (session_id, start_time, 'running', offset))
        conn.commit()
        conn.close()
        
        print(f"Starting from offset: {offset:,}")
        
        while True:
            if max_records and session_records >= max_records:
                print(f"Reached maximum records limit: {max_records:,}")
                break
                
            params = {
                'country': 'ID',  # Indonesia
                'kingdom': 'Animalia',
                'hasCoordinate': 'true',
                'hasGeospatialIssue': 'false',
                'limit': batch_size,
                'offset': offset
            }
            
            try:
                print(f"API Call #{total_api_calls + 1} - Fetching records {offset:,} to {offset + batch_size:,}")
                response = requests.get(url, params=params, timeout=60)
                total_api_calls += 1
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'results' in data and data['results']:
                        batch_results = data['results']
                        new_records = 0
                        
                        # Filter out existing records
                        for record in batch_results:
                            gbif_id = record.get('gbifID')
                            if gbif_id and gbif_id not in existing_ids:
                                all_results.append(record)
                                existing_ids.add(gbif_id)
                                new_records += 1
                        
                        session_records += new_records
                        offset += batch_size
                        
                        print(f"  -> Got {len(batch_results)} records, {new_records} new")
                        print(f"  -> Session total: {session_records:,} new records")
                        print(f"  -> Total in memory: {len(all_results):,}")
                        
                        # Save batch to database every 10 batches (3000 records)
                        if len(all_results) >= 3000:
                            self.save_batch_to_database(all_results, session_id)
                            all_results = []  # Clear memory
                            
                        # Update metadata
                        self.crawl_metadata['last_offset'] = offset
                        self.crawl_metadata['total_records'] = existing_count + session_records
                        
                        # Save metadata every 5 batches
                        if total_api_calls % 5 == 0:
                            self.save_crawl_metadata()
                            self.update_session_in_database(session_id, offset, session_records)
                        
                        # Check if we've got all available data
                        if len(batch_results) < batch_size:
                            print("Reached end of available data")
                            break
                            
                        # Rate limiting - be nice to GBIF
                        time.sleep(0.2)  # 200ms delay between requests
                        
                    else:
                        print("No more results available")
                        break
                        
                elif response.status_code == 414:
                    print("Request URI too long, trying with smaller parameters")
                    break
                elif response.status_code == 429:
                    print("Rate limited, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                else:
                    print(f"API request failed with status code: {response.status_code}")
                    print(f"Response: {response.text[:500]}")
                    if response.status_code >= 500:
                        print("Server error, waiting 30 seconds...")
                        time.sleep(30)
                        continue
                    else:
                        break
                    
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                print("Waiting 30 seconds before retry...")
                time.sleep(30)
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                break
        
        # Save remaining results
        if all_results:
            self.save_batch_to_database(all_results, session_id)
        
        # Finalize crawl session
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.crawl_metadata['end_time'] = end_time
        self.crawl_metadata['total_records'] = existing_count + session_records
        
        # Update session in database
        self.finalize_session_in_database(session_id, session_records, 'completed')
        
        print(f"\n{'='*60}")
        print(f"CRAWL SESSION COMPLETED")
        print(f"{'='*60}")
        print(f"Session ID: {session_id}")
        print(f"Duration: {duration}")
        print(f"New records collected: {session_records:,}")
        print(f"Total API calls made: {total_api_calls:,}")
        print(f"Total records in database: {self.get_existing_record_count():,}")
        print(f"Average records per API call: {session_records/max(1, total_api_calls):.1f}")
        
        self.save_crawl_metadata()
        return session_records
        
    def save_batch_to_database(self, records, session_id):
        """Save a batch of records to database"""
        if not records:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        records_to_insert = []
        for record in records:
            # Extract and clean data
            record_data = (
                record.get('gbifID'),
                record.get('species'),
                record.get('scientificName'),
                record.get('kingdom'),
                record.get('phylum'),
                record.get('class'),
                record.get('order'),  # Note: 'order' is a reserved word in SQL, using order_name
                record.get('family'),
                record.get('genus'),
                record.get('decimalLatitude'),
                record.get('decimalLongitude'),
                record.get('country'),
                record.get('stateProvince'),
                record.get('locality'),
                record.get('year'),
                record.get('month'),
                record.get('day'),
                record.get('basisOfRecord'),
                record.get('institutionCode'),
                record.get('collectionCode'),
                record.get('catalogNumber'),
                record.get('recordedBy'),
                record.get('eventDate'),
                record.get('coordinateUncertaintyInMeters'),
                record.get('elevation'),
                record.get('depth'),
                session_id
            )
            records_to_insert.append(record_data)
        
        # Bulk insert
        cursor.executemany('''
            INSERT OR IGNORE INTO occurrences (
                gbifID, species, scientificName, kingdom, phylum, class, order_name,
                family, genus, decimalLatitude, decimalLongitude, country, stateProvince,
                locality, year, month, day, basisOfRecord, institutionCode, collectionCode,
                catalogNumber, recordedBy, eventDate, coordinateUncertaintyInMeters,
                elevation, depth, crawl_session
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', records_to_insert)
        
        conn.commit()
        conn.close()
        
        print(f"  -> Saved {len(records_to_insert)} records to database")
        
    def update_session_in_database(self, session_id, offset, records_count):
        """Update session progress in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE crawl_metadata 
            SET last_offset = ?, total_records = ?
            WHERE session_id = ?
        ''', (offset, records_count, session_id))
        conn.commit()
        conn.close()
        
    def finalize_session_in_database(self, session_id, total_records, status):
        """Finalize session in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE crawl_metadata 
            SET end_time = ?, total_records = ?, status = ?
            WHERE session_id = ?
        ''', (datetime.now(), total_records, status, session_id))
        conn.commit()
        conn.close()
        
    def load_all_data_from_database(self):
        """Load all occurrence data from database into memory"""
        print("Loading all data from database...")
        
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT * FROM occurrences 
            WHERE decimalLatitude IS NOT NULL 
            AND decimalLongitude IS NOT NULL
            AND species IS NOT NULL
        '''
        
        self.occurrence_data = pd.read_sql_query(query, conn)
        conn.close()
        
        # Filter by Indonesia bounds
        self.occurrence_data = self.occurrence_data[
            (self.occurrence_data['decimalLatitude'].between(
                self.indonesia_bounds['min_lat'], 
                self.indonesia_bounds['max_lat']
            )) &
            (self.occurrence_data['decimalLongitude'].between(
                self.indonesia_bounds['min_lng'], 
                self.indonesia_bounds['max_lng']
            ))
        ]
        
        print(f"Loaded {len(self.occurrence_data):,} valid occurrence records")
        return self.occurrence_data
        
    def get_database_statistics(self):
        """Get comprehensive statistics about the database"""
        conn = sqlite3.connect(self.db_path)
        
        stats = {}
        
        # Total records
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM occurrences")
        stats['total_records'] = cursor.fetchone()[0]
        
        # Records with coordinates
        cursor.execute("SELECT COUNT(*) FROM occurrences WHERE decimalLatitude IS NOT NULL AND decimalLongitude IS NOT NULL")
        stats['records_with_coordinates'] = cursor.fetchone()[0]
        
        # Unique species
        cursor.execute("SELECT COUNT(DISTINCT species) FROM occurrences WHERE species IS NOT NULL")
        stats['unique_species'] = cursor.fetchone()[0]
        
        # Records by class
        cursor.execute("""
            SELECT class, COUNT(*) as count 
            FROM occurrences 
            WHERE class IS NOT NULL 
            GROUP BY class 
            ORDER BY count DESC 
            LIMIT 10
        """)
        stats['top_classes'] = cursor.fetchall()
        
        # Records by province
        cursor.execute("""
            SELECT stateProvince, COUNT(*) as count 
            FROM occurrences 
            WHERE stateProvince IS NOT NULL 
            GROUP BY stateProvince 
            ORDER BY count DESC 
            LIMIT 15
        """)
        stats['top_provinces'] = cursor.fetchall()
        
        # Crawl sessions
        cursor.execute("SELECT session_id, start_time, end_time, total_records, status FROM crawl_metadata ORDER BY start_time DESC")
        stats['crawl_sessions'] = cursor.fetchall()
        
        conn.close()
        return stats
        
    def print_database_statistics(self):
        """Print comprehensive database statistics"""
        stats = self.get_database_statistics()
        
        print(f"\n{'='*60}")
        print(f"DATABASE STATISTICS")
        print(f"{'='*60}")
        print(f"Total Records: {stats['total_records']:,}")
        print(f"Records with Coordinates: {stats['records_with_coordinates']:,}")
        print(f"Unique Species: {stats['unique_species']:,}")
        
        print(f"\n{'='*40}")
        print(f"TOP TAXONOMIC CLASSES")
        print(f"{'='*40}")
        for class_name, count in stats['top_classes']:
            print(f"{class_name or 'Unknown'}: {count:,} records")
            
        print(f"\n{'='*40}")
        print(f"TOP PROVINCES BY RECORD COUNT")
        print(f"{'='*40}")
        for province, count in stats['top_provinces']:
            print(f"{province or 'Unknown'}: {count:,} records")
            
        print(f"\n{'='*40}")
        print(f"CRAWL SESSIONS")
        print(f"{'='*40}")
        for session_id, start_time, end_time, total_records, status in stats['crawl_sessions']:
            print(f"{session_id}: {total_records:,} records ({status})")
            
    def calculate_shannon_index(self, species_counts):
        """Calculate Shannon Diversity Index"""
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
        """Calculate Shannon Index for each province using database data"""
        if self.occurrence_data is None:
            self.load_all_data_from_database()
        
        if self.occurrence_data is None or len(self.occurrence_data) == 0:
            print("No occurrence data available")
            return None
        
        print("Analyzing biodiversity by province...")
        province_biodiversity = {}
        
        # Group by province
        provinces = self.occurrence_data['stateProvince'].dropna().unique()
        print(f"Analyzing {len(provinces)} provinces...")
        
        for i, province in enumerate(provinces, 1):
            print(f"Processing province {i}/{len(provinces)}: {province}")
            
            province_data = self.occurrence_data[
                self.occurrence_data['stateProvince'] == province
            ]
            
            # Count species occurrences
            species_counts = Counter(province_data['species'].dropna())
            
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
            
            # Get taxonomic diversity
            class_counts = Counter(province_data['class'].dropna())
            family_counts = Counter(province_data['family'].dropna())
            
            province_biodiversity[province] = {
                'shannon_index': shannon_index,
                'species_richness': species_richness,
                'total_occurrences': total_occurrences,
                'evenness': evenness,
                'dominant_species': species_counts.most_common(10),
                'class_diversity': len(class_counts),
                'family_diversity': len(family_counts),
                'top_classes': class_counts.most_common(5),
                'top_families': family_counts.most_common(5)
            }
        
        self.province_data = province_biodiversity
        print(f"Completed biodiversity analysis for {len(provinces)} provinces")
        return province_biodiversity
    
    def create_interactive_map(self):
        """Create comprehensive interactive map"""
        if self.province_data is None:
            print("No province biodiversity data available")
            return None
        
        print("Creating comprehensive interactive map...")
        
        # Create base map centered on Indonesia
        m = folium.Map(
            location=[-2.5, 118.0],
            zoom_start=5,
            tiles='OpenStreetMap'
        )
        
        # Add different tile layers
        folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)
        folium.TileLayer('Stamen Terrain', name='Terrain').add_to(m)
        
        # Add heatmap of species occurrences
        if self.occurrence_data is not None and len(self.occurrence_data) > 0:
            # Sample data for heatmap if too large
            sample_size = min(10000, len(self.occurrence_data))
            sample_data = self.occurrence_data.sample(n=sample_size)
            
            heat_data = []
            for _, row in sample_data.iterrows():
                if not pd.isna(row['decimalLatitude']) and not pd.isna(row['decimalLongitude']):
                    heat_data.append([row['decimalLatitude'], row['decimalLongitude']])
            
            if heat_data:
                plugins.HeatMap(
                    heat_data, 
                    name='Species Occurrences Heatmap', 
                    show=False,
                    radius=8,
                    blur=10
                ).add_to(m)
        
        # Add province markers with comprehensive popups
        for province, data in self.province_data.items():
            province_coords = self.occurrence_data[
                self.occurrence_data['stateProvince'] == province
            ][['decimalLatitude', 'decimalLongitude']].mean()
            
            if not province_coords.isna().any():
                # Create comprehensive popup
                popup_html = f"""
                <div style="width: 450px; font-family: Arial, sans-serif; max-height: 500px; overflow-y: auto;">
                    <h3 style="color: #2E8B57; margin-bottom: 10px; text-align: center; border-bottom: 2px solid #2E8B57; padding-bottom: 5px;">{province}</h3>
                    
                    <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <h4 style="margin: 0 0 8px 0; color: #1e4d72;">Biodiversity Metrics</h4>
                        <table style="width: 100%; font-size: 12px;">
                            <tr><td><b>Shannon Index:</b></td><td>{data['shannon_index']:.3f}</td></tr>
                            <tr><td><b>Species Richness:</b></td><td>{data['species_richness']:,}</td></tr>
                            <tr><td><b>Evenness Index:</b></td><td>{data['evenness']:.3f}</td></tr>
                            <tr><td><b>Total Records:</b></td><td>{data['total_occurrences']:,}</td></tr>
                            <tr><td><b>Class Diversity:</b></td><td>{data['class_diversity']}</td></tr>
                            <tr><td><b>Family Diversity:</b></td><td>{data['family_diversity']}</td></tr>
                        </table>
                    </div>
                    
                    <div style="background-color: #fff8dc; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <h4 style="margin: 0 0 8px 0; color: #8B4513;">Top 5 Species</h4>
                        <ol style="margin: 0; padding-left: 20px; font-size: 11px;">
                """
                
                for species, count in data['dominant_species'][:5]:
                    popup_html += f"<li><i>{species}</i>: {count:,} records</li>"
                
                popup_html += """
                        </ol>
                    </div>
                    
                    <div style="background-color: #f5f5dc; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <h4 style="margin: 0 0 8px 0; color: #8B4513;">Top Taxonomic Classes</h4>
                        <ul style="margin: 0; padding-left: 20px; font-size: 11px;">
                """
                
                for class_name, count in data['top_classes'][:5]:
                    popup_html += f"<li>{class_name or 'Unknown'}: {count:,} records</li>"
                
                popup_html += """
                        </ul>
                    </div>
                    
                    <div style="text-align: center; margin-top: 10px; font-size: 10px; color: #666;">
                        Click outside to close | Data from GBIF
                    </div>
                </div>
                """
                
                # Determine marker color and size based on Shannon Index
                shannon_index = data['shannon_index']
                if shannon_index > 3.0:
                    color = '#003d00'  # Very dark green
                    size_multiplier = 8
                elif shannon_index > 2.5:
                    color = '#006400'  # Dark green
                    size_multiplier = 7
                elif shannon_index > 2.0:
                    color = '#32CD32'  # Lime green
                    size_multiplier = 6
                elif shannon_index > 1.5:
                    color = '#FFA500'  # Orange
                    size_multiplier = 5
                elif shannon_index > 1.0:
                    color = '#FF6347'  # Tomato
                    size_multiplier = 4
                else:
                    color = '#FF0000'  # Red
                    size_multiplier = 3
                
                radius = max(8, min(25, shannon_index * size_multiplier))
                
                # Create marker
                folium.CircleMarker(
                    location=[province_coords['decimalLatitude'], province_coords['decimalLongitude']],
                    radius=radius,
                    popup=folium.Popup(popup_html, max_width=500, max_height=400),
                    color='white',
                    weight=2,
                    fillColor=color,
                    fillOpacity=0.8,
                    tooltip=f"{province}<br>Shannon: {shannon_index:.3f}<br>Species: {data['species_richness']:,}"
                ).add_to(m)
        
        # Enhanced legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 280px; height: 280px; 
                    background-color: white; border: 2px solid grey; z-index: 9999; 
                    font-size: 12px; padding: 15px; border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            
            <h4 style="margin: 0 0 10px 0; color: #2E8B57; text-align: center;">Shannon Diversity Index</h4>
            
            <div style="margin-bottom: 8px;">
                <span style="display: inline-block; width: 15px; height: 15px; 
                           background-color: #003d00; border-radius: 50%; margin-right: 8px;"></span>
                Exceptional (> 3.0)
            </div>
            
            <div style="margin-bottom: 8px;">
                <span style="display: inline-block; width: 15px; height: 15px; 
                           background-color: #006400; border-radius: 50%; margin-right: 8px;"></span>
                Very High (2.5 - 3.0)
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
            
            <div style="margin-bottom: 8px;">
                <span style="display: inline-block; width: 15px; height: 15px; 
                           background-color: #FF6347; border-radius: 50%; margin-right: 8px;"></span>
                Low (1.0 - 1.5)
            </div>
            
            <div style="margin-bottom: 12px;">
                <span style="display: inline-block; width: 15px; height: 15px; 
                           background-color: #FF0000; border-radius: 50%; margin-right: 8px;"></span>
                Very Low (< 1.0)
            </div>
            
            <hr style="margin: 10px 0;">
            
            <div style="font-size: 10px; color: #666;">
                • Circle size ∝ Shannon Index<br>
                • Click markers for detailed info<br>
                • Toggle layers in top-right<br>
                • Heatmap shows occurrence density<br>
                • Based on comprehensive GBIF data
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Add measure tool
        plugins.MeasureControl().add_to(m)
        
        return m
    
    def export_comprehensive_results(self, filename_prefix="indonesia_comprehensive_biodiversity"):
        """Export comprehensive results in multiple formats"""
        if not self.province_data:
            print("No data to export")
            return None
        
        print("Exporting comprehensive results...")
        
        # Create main DataFrame for export
        export_data = []
        for province, data in self.province_data.items():
            row = {
                'Province': province,
                'Shannon_Index': data['shannon_index'],
                'Species_Richness': data['species_richness'],
                'Total_Occurrences': data['total_occurrences'],
                'Evenness_Index': data['evenness'],
                'Class_Diversity': data['class_diversity'],
                'Family_Diversity': data['family_diversity'],
            }
            
            # Add top species
            for i, (species, count) in enumerate(data['dominant_species'][:5], 1):
                row[f'Top_Species_{i}'] = species
                row[f'Top_Species_{i}_Count'] = count
            
            # Add top classes
            for i, (class_name, count) in enumerate(data['top_classes'][:3], 1):
                row[f'Top_Class_{i}'] = class_name
                row[f'Top_Class_{i}_Count'] = count
            
            export_data.append(row)
        
        df = pd.DataFrame(export_data)
        
        # Export to multiple formats
        csv_filename = f"{filename_prefix}_results.csv"
        df.to_csv(csv_filename, index=False)
        
        excel_filename = f"{filename_prefix}_results.xlsx"
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Province_Summary', index=False)
            
            # Add detailed species data
            if self.occurrence_data is not None:
                species_summary = self.occurrence_data.groupby('species').agg({
                    'gbifID': 'count',
                    'stateProvince': 'nunique',
                    'class': 'first',
                    'family': 'first'
                }).rename(columns={
                    'gbifID': 'Total_Records',
                    'stateProvince': 'Provinces_Found'
                }).sort_values('Total_Records', ascending=False)
                
                species_summary.to_excel(writer, sheet_name='Species_Summary')
        
        # Export database statistics
        stats = self.get_database_statistics()
        stats_filename = f"{filename_prefix}_database_stats.json"
        with open(stats_filename, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"✅ Exported results:")
        print(f"   - {csv_filename}")
        print(f"   - {excel_filename}")
        print(f"   - {stats_filename}")
        
        return df
    
    def generate_comprehensive_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        if self.province_data is None:
            return None
        
        # Load fresh database statistics
        stats = self.get_database_statistics()
        
        # Create DataFrame for analysis
        summary_df = pd.DataFrame([
            {
                'Province': province,
                'Shannon_Index': data['shannon_index'],
                'Species_Richness': data['species_richness'],
                'Total_Occurrences': data['total_occurrences'],
                'Evenness': data['evenness'],
                'Class_Diversity': data['class_diversity'],
                'Family_Diversity': data['family_diversity']
            }
            for province, data in self.province_data.items()
        ])
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE INDONESIAN ANIMALIA BIODIVERSITY ANALYSIS")
        print(f"{'='*80}")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Database Records: {stats['total_records']:,}")
        print(f"Records with Coordinates: {stats['records_with_coordinates']:,}")
        print(f"Unique Species Documented: {stats['unique_species']:,}")
        print(f"Provinces Analyzed: {len(summary_df)}")
        
        print(f"\n{'='*50}")
        print(f"SHANNON DIVERSITY INDEX STATISTICS")
        print(f"{'='*50}")
        print(f"Mean Shannon Index: {summary_df['Shannon_Index'].mean():.4f}")
        print(f"Median Shannon Index: {summary_df['Shannon_Index'].median():.4f}")
        print(f"Standard Deviation: {summary_df['Shannon_Index'].std():.4f}")
        print(f"Maximum Shannon Index: {summary_df['Shannon_Index'].max():.4f}")
        print(f"Minimum Shannon Index: {summary_df['Shannon_Index'].min():.4f}")
        
        print(f"\n{'='*50}")
        print(f"SPECIES RICHNESS STATISTICS")
        print(f"{'='*50}")
        print(f"Mean Species per Province: {summary_df['Species_Richness'].mean():.1f}")
        print(f"Total Unique Species: {summary_df['Species_Richness'].sum():,}")
        print(f"Most Species-Rich Province: {summary_df.loc[summary_df['Species_Richness'].idxmax(), 'Province']} ({summary_df['Species_Richness'].max():,} species)")
        
        # Top 10 most diverse provinces
        print(f"\n{'='*50}")
        print(f"TOP 10 MOST BIODIVERSE PROVINCES (Shannon Index)")
        print(f"{'='*50}")
        top_provinces = summary_df.nlargest(10, 'Shannon_Index')
        for i, (_, row) in enumerate(top_provinces.iterrows(), 1):
            print(f"{i:2d}. {row['Province']:<25} Shannon: {row['Shannon_Index']:.4f} | Species: {row['Species_Richness']:,}")
        
        # Bottom 5 provinces
        print(f"\n{'='*50}")
        print(f"PROVINCES WITH LOWEST DIVERSITY")
        print(f"{'='*50}")
        bottom_provinces = summary_df.nsmallest(5, 'Shannon_Index')
        for i, (_, row) in enumerate(bottom_provinces.iterrows(), 1):
            print(f"{i}. {row['Province']:<25} Shannon: {row['Shannon_Index']:.4f} | Species: {row['Species_Richness']:,}")
        
        # Taxonomic diversity
        print(f"\n{'='*50}")
        print(f"TAXONOMIC DIVERSITY OVERVIEW")
        print(f"{'='*50}")
        print("Top Taxonomic Classes in Database:")
        for class_name, count in stats['top_classes'][:10]:
            print(f"  {class_name or 'Unknown':<20}: {count:,} records")
        
        # Data quality metrics
        coord_coverage = (stats['records_with_coordinates'] / stats['total_records']) * 100 if stats['total_records'] > 0 else 0
        print(f"\n{'='*50}")
        print(f"DATA QUALITY METRICS")
        print(f"{'='*50}")
        print(f"Geographic Coverage: {coord_coverage:.1f}% of records have coordinates")
        print(f"Average Records per Species: {stats['total_records'] / max(1, stats['unique_species']):.1f}")
        
        # Crawl session summary
        print(f"\n{'='*50}")
        print(f"DATA COLLECTION SESSIONS")
        print(f"{'='*50}")
        for session_id, start_time, end_time, total_records, status in stats['crawl_sessions'][-5:]:  # Last 5 sessions
            print(f"Session: {session_id}")
            print(f"  Status: {status} | Records: {total_records:,}")
            print(f"  Time: {start_time} - {end_time}")
        
        return summary_df

def run_comprehensive_analysis():
    """
    Complete workflow for comprehensive biodiversity analysis
    """
    print("=== COMPREHENSIVE INDONESIAN ANIMALIA BIODIVERSITY ANALYSIS ===\n")
    print("This analysis will attempt to crawl ALL available animal occurrence data")
    print("from GBIF for Indonesia and perform comprehensive biodiversity analysis.\n")
    
    # Initialize analyzer
    analyzer = ComprehensiveBiodiversityAnalyzer()
    
    # Print current database status
    analyzer.print_database_statistics()
    
    # Ask user if they want to crawl more data
    existing_count = analyzer.get_existing_record_count()
    
    if existing_count > 0:
        print(f"\nFound {existing_count:,} existing records in database.")
        crawl_more = input("Do you want to crawl additional data? (y/n): ").lower().strip()
        
        if crawl_more == 'y':
            max_new_records = input("Enter maximum new records to crawl (or press Enter for unlimited): ").strip()
            max_new_records = int(max_new_records) if max_new_records.isdigit() else None
            
            print(f"\nStarting comprehensive crawl...")
            print("Note: This may take several hours to complete for all available data.")
            print("The process is resumable - you can stop and restart anytime.")
            
            # Start crawling
            new_records = analyzer.comprehensive_gbif_crawl(max_records=max_new_records, resume=True)
            print(f"\nCrawl completed! Added {new_records:,} new records.")
        else:
            print("Using existing data for analysis...")
    else:
        print("No existing data found. Starting fresh crawl...")
        print("Note: This will take several hours to crawl all available data.")
        
        # Start crawling
        new_records = analyzer.comprehensive_gbif_crawl(max_records=50000, resume=False)  # Start with reasonable limit
        print(f"\nInitial crawl completed! Collected {new_records:,} records.")
    
    # Load data and analyze
    print("\nStep 1: Loading data from database...")
    data = analyzer.load_all_data_from_database()
    
    if data is None or len(data) == 0:
        print("No valid data available for analysis")
        return None, None
    
    print(f"Loaded {len(data):,} occurrence records for analysis")
    
    # Analyze biodiversity
    print("\nStep 2: Calculating Shannon indices for each province...")
    biodiversity_results = analyzer.analyze_biodiversity_by_province()
    
    if biodiversity_results is None:
        print("Failed to analyze biodiversity")
        return None, None
    
    # Generate comprehensive summary statistics
    print("\nStep 3: Generating comprehensive summary statistics...")
    summary_stats = analyzer.generate_comprehensive_summary_statistics()
    
    # Create interactive map
    print("\nStep 4: Creating comprehensive interactive map...")
    biodiversity_map = analyzer.create_interactive_map()
    
    if biodiversity_map:
        map_filename = "indonesia_comprehensive_biodiversity_map.html"
        biodiversity_map.save(map_filename)
        print(f"\n✅ Comprehensive interactive map saved as '{map_filename}'")
        print("   This map includes detailed popups with taxonomic information!")
    
    # Export comprehensive results
    print("\nStep 5: Exporting comprehensive results...")
    export_df = analyzer.export_comprehensive_results()
    
    # Print final database statistics
    print("\nStep 6: Final database summary...")
    analyzer.print_database_statistics()
    
    # Final summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS COMPLETE! 🎉")
    print("="*80)
    print("Files generated:")
    print("1. indonesia_comprehensive_biodiversity_map.html - Interactive map with detailed province data")
    print("2. indonesia_comprehensive_biodiversity_results.csv - Summary results")
    print("3. indonesia_comprehensive_biodiversity_results.xlsx - Detailed Excel workbook")
    print("4. indonesia_comprehensive_biodiversity_database_stats.json - Database statistics")
    print("5. biodiversity_data/indonesia_biodiversity.db - SQLite database with all records")
    
    print(f"\n📊 Key Findings:")
    if summary_stats is not None:
        total_species = analyzer.get_database_statistics()['unique_species']
        total_records = analyzer.get_database_statistics()['total_records']
        avg_shannon = summary_stats['Shannon_Index'].mean()
        most_diverse = summary_stats.loc[summary_stats['Shannon_Index'].idxmax(), 'Province']
        
        print(f"- Total unique animal species documented: {total_species:,}")
        print(f"- Total occurrence records analyzed: {total_records:,}")
        print(f"- Average Shannon diversity index: {avg_shannon:.4f}")
        print(f"- Most biodiverse province: {most_diverse}")
    
    print(f"\n🔬 For your research:")
    print("- The SQLite database contains all raw occurrence data for further analysis")
    print("- Export custom queries from the database for specific taxonomic groups")
    print("- Use the Excel file for statistical analysis and visualization")
    print("- Reference the comprehensive methodology in your paper")
    print("- The interactive map provides visual evidence for geographic patterns")
    
    return analyzer, export_df

# Run the comprehensive analysis
if __name__ == "__main__":
    analyzer, results = run_comprehensive_analysis()
    
    # Optional: Example of how to query the database for specific analyses
    if analyzer:
        print(f"\n{'='*60}")
        print("EXAMPLE: Additional Analysis Capabilities")
        print("="*60)
        print("The SQLite database allows for custom queries. Examples:")
        print("1. Species found in multiple provinces")
        print("2. Temporal patterns in data collection")
        print("3. Taxonomic group analysis (mammals, birds, etc.)")
        print("4. Coordinate-based spatial analysis")
        print("5. Institution-based data quality assessment")
        
        # Example query
        try:
            import sqlite3
            conn = sqlite3.connect(analyzer.db_path)
            cursor = conn.cursor()
            
            # Get most widespread species
            cursor.execute("""
                SELECT species, COUNT(DISTINCT stateProvince) as provinces_found, COUNT(*) as total_records
                FROM occurrences 
                WHERE species IS NOT NULL AND stateProvince IS NOT NULL
                GROUP BY species 
                ORDER BY provinces_found DESC, total_records DESC 
                LIMIT 10
            """)
            
            widespread_species = cursor.fetchall()
            conn.close()
            
            print(f"\nTop 10 Most Widespread Species (found in most provinces):")
            for i, (species, provinces, records) in enumerate(widespread_species, 1):
                print(f"{i:2d}. {species}")
                print(f"    Found in {provinces} provinces, {records:,} total records")
                
        except Exception as e:
            print(f"Database query example failed: {e}")
    
    print(f"\n{'='*60}")
    print("Analysis complete! Check the generated files for your research.")
    print("="*60)