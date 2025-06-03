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
    def __init__(self, data_dir="biodiversity_data", gbif_username=None, gbif_password=None, gbif_email=None):
        self.indonesia_bounds = {
            'min_lat': -11.0,
            'max_lat': 6.0,
            'min_lng': 95.0,
            'max_lng': 141.0
        }
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # GBIF credentials for bulk downloads
        self.gbif_username = gbif_username
        self.gbif_password = gbif_password
        self.gbif_email = gbif_email
        
        # Database setup for efficient storage and querying
        self.db_path = self.data_dir / "indonesia_biodiversity.db"
        self.setup_database()
        
        self.occurrence_data = None
        self.province_data = None
        self.download_metadata = {
            'downloads': [],
            'total_records': 0,
            'last_download_key': None
        }
        
        # Fallback crawl metadata for API crawling
        self.crawl_metadata = {
            'total_records': 0,
            'last_offset': 0,
            'start_time': None,
            'end_time': None,
            'crawl_sessions': []
        }
        
        # Load existing metadata if available
        self.load_download_metadata()
        
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
        
    def load_download_metadata(self):
        """Load existing download metadata"""
        metadata_file = self.data_dir / "download_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.download_metadata = json.load(f)
        
        # Also load crawl metadata for fallback
        crawl_metadata_file = self.data_dir / "crawl_metadata.json"
        if crawl_metadata_file.exists():
            with open(crawl_metadata_file, 'r') as f:
                self.crawl_metadata = json.load(f)
                
    def save_download_metadata(self):
        """Save download metadata"""
        metadata_file = self.data_dir / "download_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.download_metadata, f, indent=2, default=str)
    
    def save_crawl_metadata(self):
        """Save crawl metadata"""
        metadata_file = self.data_dir / "crawl_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.crawl_metadata, f, indent=2, default=str)
            
    def request_gbif_bulk_download(self):
        """
        Request bulk download of all Indonesian animal data using GBIF Download API
        """
        if not all([self.gbif_username, self.gbif_password, self.gbif_email]):
            print("❌ GBIF credentials required for bulk downloads!")
            print("\n🔐 To use bulk downloads, you need:")
            print("1. Register at: https://www.gbif.org/user/profile")
            print("2. Get your username, password, and email")
            print("3. Initialize with: analyzer = ComprehensiveBiodiversityAnalyzer(")
            print("     gbif_username='your_username',")
            print("     gbif_password='your_password',")
            print("     gbif_email='your@email.com')")
            print("\n⚠️  Falling back to API crawling method...")
            return None
            
        print("🚀 Requesting GBIF bulk download for Indonesian animals...")
        
        # Create download request
        download_request = {
            "creator": self.gbif_username,
            "notificationAddresses": [self.gbif_email],
            "sendNotification": True,
            "format": "SIMPLE_CSV",
            "predicate": {
                "type": "and",
                "predicates": [
                    {
                        "type": "equals",
                        "key": "COUNTRY",
                        "value": "ID"  # Indonesia
                    },
                    {
                        "type": "equals",
                        "key": "KINGDOM_KEY",  # Fixed: use KINGDOM_KEY instead of KINGDOM
                        "value": "1"  # Animalia kingdom key in GBIF
                    },
                    {
                        "type": "equals",
                        "key": "HAS_COORDINATE",
                        "value": "true"
                    },
                    {
                        "type": "equals",
                        "key": "HAS_GEOSPATIAL_ISSUE",
                        "value": "false"
                    }
                ]
            }
        }
        
        # Submit download request
        url = "https://api.gbif.org/v1/occurrence/download/request"
        
        try:
            response = requests.post(
                url,
                json=download_request,
                auth=(self.gbif_username, self.gbif_password),
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 201:
                download_key = response.text.strip().strip('"')
                print(f"✅ Download request submitted successfully!")
                print(f"📋 Download key: {download_key}")
                print(f"📧 You'll receive an email at {self.gbif_email} when ready")
                
                # Save download info
                download_info = {
                    'download_key': download_key,
                    'request_time': datetime.now(),
                    'status': 'RUNNING',
                    'request': download_request
                }
                
                self.download_metadata['downloads'].append(download_info)
                self.download_metadata['last_download_key'] = download_key
                self.save_download_metadata()
                
                return download_key
                
            else:
                print(f"❌ Download request failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error submitting download request: {e}")
            return None
    
    def check_download_status(self, download_key):
        """Check the status of a GBIF download"""
        url = f"https://api.gbif.org/v1/occurrence/download/{download_key}"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                download_info = response.json()
                return download_info
            else:
                print(f"Error checking download status: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error checking download status: {e}")
            return None
    
    def wait_for_download_completion(self, download_key, check_interval=60):
        """
        Wait for download to complete, checking status periodically
        """
        print(f"⏳ Waiting for download {download_key} to complete...")
        print(f"🔄 Checking every {check_interval} seconds...")
        
        start_time = time.time()
        
        while True:
            status_info = self.check_download_status(download_key)
            
            if status_info:
                status = status_info.get('status')
                total_records = status_info.get('totalRecords', 0)
                
                elapsed = int(time.time() - start_time)
                print(f"📊 Status: {status} | Records: {total_records:,} | Elapsed: {elapsed//60}m {elapsed%60}s")
                
                if status == 'SUCCEEDED':
                    download_link = status_info.get('downloadLink')
                    size_mb = status_info.get('size', 0) / (1024 * 1024)
                    
                    print(f"✅ Download completed!")
                    print(f"📁 File size: {size_mb:.1f} MB")
                    print(f"🔗 Download link: {download_link}")
                    print(f"🏷️  DOI: {status_info.get('doi', 'Not assigned')}")
                    
                    # Update metadata
                    for download in self.download_metadata['downloads']:
                        if download['download_key'] == download_key:
                            download.update({
                                'status': 'SUCCEEDED',
                                'completion_time': datetime.now(),
                                'total_records': total_records,
                                'download_link': download_link,
                                'size_bytes': status_info.get('size', 0),
                                'doi': status_info.get('doi')
                            })
                            break
                    
                    self.save_download_metadata()
                    return download_link
                    
                elif status == 'FAILED':
                    print(f"❌ Download failed!")
                    return None
                    
                elif status == 'CANCELLED':
                    print(f"⚠️  Download was cancelled!")
                    return None
                    
                else:  # RUNNING, PREPARING
                    time.sleep(check_interval)
                    
            else:
                print("❌ Could not check download status")
                time.sleep(check_interval)
    
    def download_and_extract_data(self, download_link):
        """
        Download and extract the GBIF data file
        """
        print(f"📥 Downloading data file...")
        
        # Download the file
        zip_filename = self.data_dir / f"gbif_indonesia_animals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        try:
            response = requests.get(download_link, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(zip_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r📥 Download progress: {progress:.1f}% ({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)", end='')
            
            print(f"\n✅ Download completed: {zip_filename}")
            
            # Extract the ZIP file
            print("📂 Extracting data...")
            import zipfile
            
            extract_dir = self.data_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                
            # Find the occurrence data file
            occurrence_files = list(extract_dir.glob("*.csv"))
            if not occurrence_files:
                occurrence_files = list(extract_dir.glob("occurrence.txt"))
                
            if occurrence_files:
                data_file = occurrence_files[0]
                print(f"📊 Found data file: {data_file}")
                return data_file
            else:
                print("❌ Could not find occurrence data file in the download")
                return None
                
        except Exception as e:
            print(f"❌ Error downloading/extracting data: {e}")
            return None
    
    def import_bulk_data_to_database(self, data_file):
        """
        Import the bulk downloaded data into our database
        """
        print(f"📥 Importing data from {data_file} into database...")
        
        try:
            # Read the CSV file in chunks to handle large files
            chunk_size = 10000
            total_imported = 0
            
            # Get file size for progress tracking
            file_size = os.path.getsize(data_file)
            processed_bytes = 0
            
            conn = sqlite3.connect(self.db_path)
            
            for chunk_num, chunk in enumerate(pd.read_csv(data_file, sep='\t', chunksize=chunk_size, low_memory=False)):
                # Clean and prepare data
                chunk = chunk.dropna(subset=['species', 'decimalLatitude', 'decimalLongitude'])
                
                # Filter by Indonesia bounds
                chunk = chunk[
                    (chunk['decimalLatitude'].between(self.indonesia_bounds['min_lat'], self.indonesia_bounds['max_lat'])) &
                    (chunk['decimalLongitude'].between(self.indonesia_bounds['min_lng'], self.indonesia_bounds['max_lng']))
                ]
                
                if len(chunk) > 0:
                    # Add download session info
                    chunk['crawl_session'] = f"bulk_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Import to database
                    chunk.to_sql('occurrences', conn, if_exists='append', index=False, method='multi')
                    total_imported += len(chunk)
                
                # Progress update
                processed_bytes += chunk_size * 1000  # Rough estimate
                if file_size > 0:
                    progress = min(100, (processed_bytes / file_size) * 100)
                    print(f"\r📥 Import progress: {progress:.1f}% | Imported: {total_imported:,} records", end='')
            
            conn.close()
            
            print(f"\n✅ Successfully imported {total_imported:,} records to database")
            self.download_metadata['total_records'] = total_imported
            self.save_download_metadata()
            
            return total_imported
            
        except Exception as e:
            print(f"❌ Error importing data: {e}")
            return 0
    
    def comprehensive_bulk_download_workflow(self):
        """
        Complete workflow using GBIF bulk download API
        """
        print("🚀 Starting comprehensive bulk download workflow...")
        
        # Step 1: Request download
        download_key = self.request_gbif_bulk_download()
        if not download_key:
            print("⚠️  Bulk download failed, falling back to API crawling...")
            return self.comprehensive_gbif_crawl(max_records=100000)
        
        # Step 2: Wait for completion
        download_link = self.wait_for_download_completion(download_key)
        if not download_link:
            print("❌ Download failed or was cancelled")
            return 0
        
        # Step 3: Download and extract
        data_file = self.download_and_extract_data(download_link)
        if not data_file:
            print("❌ Could not download or extract data file")
            return 0
        
        # Step 4: Import to database
        imported_count = self.import_bulk_data_to_database(data_file)
        
        print(f"\n🎉 Bulk download workflow completed!")
        print(f"📊 Total records imported: {imported_count:,}")
        
        return imported_count
            
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
    Complete workflow for comprehensive biodiversity analysis using bulk downloads
    """
    print("=== COMPREHENSIVE INDONESIAN ANIMALIA BIODIVERSITY ANALYSIS ===\n")
    print("This analysis will download ALL available animal occurrence data")
    print("from GBIF for Indonesia using the efficient bulk download API.\n")
    
    # Get GBIF credentials
    print("🔐 GBIF Account Setup Required")
    print("="*40)
    print("To download all Indonesian animal data, you need a free GBIF account:")
    print("1. Register at: https://www.gbif.org/user/profile")
    print("2. Verify your email")
    print("3. Enter your credentials below\n")
    
    gbif_username = input("Enter your GBIF username: ").strip()
    gbif_password = input("Enter your GBIF password: ").strip()
    gbif_email = input("Enter your GBIF email: ").strip()
    
    if not all([gbif_username, gbif_password, gbif_email]):
        print("❌ All credentials are required for bulk downloads!")
        print("🔄 Falling back to API crawling with sample data...")
        
        # Initialize analyzer without credentials (will use sample data)
        analyzer = ComprehensiveBiodiversityAnalyzer()
        data = analyzer.get_sample_data()
        
    else:
        # Initialize analyzer with credentials
        analyzer = ComprehensiveBiodiversityAnalyzer(
            gbif_username=gbif_username,
            gbif_password=gbif_password, 
            gbif_email=gbif_email
        )
        
        # Check existing data
        existing_count = analyzer.get_existing_record_count()
        if existing_count > 0:
            print(f"\n📊 Found {existing_count:,} existing records in database.")
            use_existing = input("Use existing data? (y/n): ").lower().strip()
            
            if use_existing != 'y':
                print("\n🚀 Starting bulk download process...")
                print("⏳ Expected time: 15-30 minutes (much faster than API crawling!)")
                
                # Run bulk download workflow
                new_records = analyzer.comprehensive_bulk_download_workflow()
                if new_records == 0:
                    print("⚠️  Bulk download failed, using existing data...")
                else:
                    print(f"✅ Successfully added {new_records:,} new records!")
            else:
                print("📊 Using existing data for analysis...")
        else:
            print("\n🚀 No existing data found. Starting bulk download...")
            print("⏳ Expected time: 15-30 minutes for complete dataset")
            print("📧 You'll receive email notification when ready")
            
            # Run bulk download workflow  
            new_records = analyzer.comprehensive_bulk_download_workflow()
            if new_records == 0:
                print("❌ Bulk download failed. Using sample data for demonstration...")
                data = analyzer.get_sample_data()
    
    # Continue with analysis...
    print("\nStep 1: Loading data from database...")
    data = analyzer.load_all_data_from_database()
    
    if data is None or len(data) == 0:
        print("No valid data available for analysis")
        return None, None
    
    print(f"Loaded {len(data):,} occurrence records for analysis")
    
    # Rest of analysis remains the same...
    print("\nStep 2: Calculating Shannon indices for each province...")
    biodiversity_results = analyzer.analyze_biodiversity_by_province()
    
    if biodiversity_results is None:
        print("Failed to analyze biodiversity")
        return None, None
    
    print("\nStep 3: Generating comprehensive summary statistics...")
    summary_stats = analyzer.generate_comprehensive_summary_statistics()
    
    print("\nStep 4: Creating comprehensive interactive map...")
    biodiversity_map = analyzer.create_interactive_map()
    
    if biodiversity_map:
        map_filename = "indonesia_comprehensive_biodiversity_map.html"
        biodiversity_map.save(map_filename)
        print(f"\n✅ Comprehensive interactive map saved as '{map_filename}'")
    
    print("\nStep 5: Exporting comprehensive results...")
    export_df = analyzer.export_comprehensive_results()
    
    print("\nStep 6: Final database summary...")
    analyzer.print_database_statistics()
    
    # Final summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS COMPLETE! 🎉")
    print("="*80)
    print("Files generated:")
    print("1. indonesia_comprehensive_biodiversity_map.html - Interactive map")
    print("2. indonesia_comprehensive_biodiversity_results.csv - Summary results")
    print("3. indonesia_comprehensive_biodiversity_results.xlsx - Detailed Excel workbook")
    print("4. indonesia_comprehensive_biodiversity_database_stats.json - Database statistics")
    print("5. biodiversity_data/indonesia_biodiversity.db - SQLite database with all records")
    
    if summary_stats is not None:
        total_species = analyzer.get_database_statistics()['unique_species']
        total_records = analyzer.get_database_statistics()['total_records']
        avg_shannon = summary_stats['Shannon_Index'].mean()
        most_diverse = summary_stats.loc[summary_stats['Shannon_Index'].idxmax(), 'Province']
        
        print(f"\n📊 Key Findings:")
        print(f"- Total unique animal species documented: {total_species:,}")
        print(f"- Total occurrence records analyzed: {total_records:,}")
        print(f"- Average Shannon diversity index: {avg_shannon:.4f}")
        print(f"- Most biodiverse province: {most_diverse}")
    
    print(f"\n🔬 For your research:")
    print("- Complete dataset of Indonesian animal biodiversity from GBIF")
    print("- All records include coordinates, taxonomy, and collection metadata")
    print("- Cite the GBIF download DOI in your paper")
    print("- Use the interactive map for visualizing geographic patterns")
    print("- Database allows custom queries for specific research questions")
    
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