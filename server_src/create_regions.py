from app import app, db, Region

def create_default_regions():
    with app.app_context():
        # Check if we need to add default regions
        if Region.query.count() == 0:
            # Add some default regions
            default_regions = [
                {
                    'name': 'North Atlantic',
                    'description': 'North Atlantic Ocean region',
                    'center_lat': 45.0,
                    'center_lng': -40.0,
                    'radius': 2000.0
                },
                {
                    'name': 'North Pacific',
                    'description': 'North Pacific Ocean region',
                    'center_lat': 40.0,
                    'center_lng': -150.0,
                    'radius': 2000.0
                },
                {
                    'name': 'South Pacific',
                    'description': 'South Pacific Ocean region',
                    'center_lat': -20.0,
                    'center_lng': -120.0,
                    'radius': 2000.0
                },
                {
                    'name': 'Indian Ocean',
                    'description': 'Indian Ocean region',
                    'center_lat': -10.0,
                    'center_lng': 80.0,
                    'radius': 2000.0
                },
                {
                    'name': 'Arctic',
                    'description': 'Arctic Ocean region',
                    'center_lat': 80.0,
                    'center_lng': 0.0,
                    'radius': 1500.0
                },
                {
                    'name': 'Southern Ocean',
                    'description': 'Southern Ocean around Antarctica',
                    'center_lat': -70.0,
                    'center_lng': 0.0,
                    'radius': 2000.0
                }
            ]
            
            for region_data in default_regions:
                region = Region(**region_data)
                db.session.add(region)
            
            db.session.commit()
            print("Added default regions to database")
        else:
            print(f"Database already has {Region.query.count()} regions")

if __name__ == "__main__":
    create_default_regions()
