import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import warnings
warnings.filterwarnings('ignore')

class HotelDemandPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.sequence_length = 30
        self.model_type = 'lstm'
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the hotel booking data with comprehensive feature engineering"""
        print("🏨 Loading Hotel Booking Demand Dataset...")
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        except FileNotFoundError:
            print("❌ Dataset file not found. Please ensure 'hotel_bookings.csv' is in the correct path.")
            return None, None, None
        
        # Display dataset overview
        print("\n📊 Dataset Overview:")
        print(f"   - Total bookings: {len(df):,}")
        print(f"   - Resort Hotel bookings: {len(df[df['hotel'] == 'Resort Hotel']):,}")
        print(f"   - City Hotel bookings: {len(df[df['hotel'] == 'City Hotel']):,}")
        print(f"   - Cancellation rate: {df['is_canceled'].mean():.2%}")
        print(f"   - Date range: {df['arrival_date_year'].min()} - {df['arrival_date_year'].max()}")
        
        # Handle missing values
        print("\n🔧 Preprocessing data...")
        df['children'] = df['children'].fillna(0)
        df['country'] = df['country'].fillna('Unknown')
        df['agent'] = df['agent'].fillna(0)
        df['company'] = df['company'].fillna(0)
        
        # Feature engineering
        print("   - Engineering features...")
        df['total_guests'] = df['adults'] + df['children'] + df['babies']
        df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        df['is_weekend_stay'] = (df['stays_in_weekend_nights'] > 0).astype(int)
        df['has_special_requests'] = (df['total_of_special_requests'] > 0).astype(int)
        df['has_parking'] = (df['required_car_parking_spaces'] > 0).astype(int)
        
        # Lead time categories
        df['booking_lead_time_category'] = pd.cut(df['lead_time'], 
                                               bins=[0, 7, 30, 90, 365, float('inf')],
                                               labels=['Last Minute', 'Short', 'Medium', 'Long', 'Very Long'])
        
        # Convert date columns to datetime
        df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + 
                                          df['arrival_date_month'] + '-' + 
                                          df['arrival_date_day_of_month'].astype(str))
        
        # Extract comprehensive temporal features
        df['arrival_month'] = df['arrival_date'].dt.month
        df['arrival_day_of_week'] = df['arrival_date'].dt.dayofweek
        df['arrival_week_of_year'] = df['arrival_date'].dt.isocalendar().week
        df['arrival_quarter'] = df['arrival_date'].dt.quarter
        df['is_peak_season'] = df['arrival_month'].isin([7, 8, 12]).astype(int)  # July, August, December
        
        # Season classification
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        df['season'] = df['arrival_month'].apply(get_season)
        
        # Select features for modeling
        self.feature_columns = [
            # Temporal features
            'lead_time', 'arrival_month', 'arrival_week_of_year', 'arrival_day_of_week',
            'arrival_quarter', 'is_peak_season',
            
            # Stay characteristics
            'stays_in_weekend_nights', 'stays_in_week_nights', 'total_nights', 'is_weekend_stay',
            
            # Guest composition
            'adults', 'children', 'babies', 'total_guests',
            
            # Customer history
            'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
            
            # Booking details
            'booking_changes', 'days_in_waiting_list', 'adr', 
            'required_car_parking_spaces', 'has_parking',
            'total_of_special_requests', 'has_special_requests'
        ]
        
        # Categorical columns to encode
        categorical_columns = [
            'hotel', 'meal', 'country', 'market_segment', 
            'distribution_channel', 'deposit_type', 'customer_type',
            'reserved_room_type', 'assigned_room_type',
            'booking_lead_time_category', 'season'
        ]
        
        # Encode categorical variables
        print("   - Encoding categorical variables...")
        for col in categorical_columns:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                self.feature_columns.append(col + '_encoded')
        
        # Prepare features and target
        # Using cancellation as target (inverse of demand confidence)
        X = df[self.feature_columns]
        y = df['is_canceled']
        
        print(f"✅ Preprocessing complete. Final feature set: {len(self.feature_columns)} features")
        print(f"   - Feature overview:")
        print(f"     * Temporal features: 8")
        print(f"     * Stay characteristics: 4")
        print(f"     * Guest composition: 4")
        print(f"     * Customer history: 3")
        print(f"     * Booking details: 7")
        print(f"     * Encoded categorical: {len(categorical_columns)}")
        
        return X, y, df
    
    def create_sequences(self, data, targets, sequence_length=30):
        """Create sequences for time series training"""
        sequences = []
        sequence_targets = []
        
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:(i + sequence_length)])
            sequence_targets.append(targets[i + sequence_length])
        
        return np.array(sequences), np.array(sequence_targets)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model for demand prediction with advanced architecture"""
        model = Sequential([
            LSTM(256, return_sequences=True, input_shape=input_shape, 
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(128, return_sequences=True, 
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(64, return_sequences=False,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu', kernel_initializer='he_normal'),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        return model
    
    def build_gru_model(self, input_shape):
        """Build GRU model alternative with similar architecture"""
        model = Sequential([
            GRU(256, return_sequences=True, input_shape=input_shape,
                kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'),
            BatchNormalization(),
            Dropout(0.3),
            
            GRU(128, return_sequences=True,
                kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'),
            BatchNormalization(),
            Dropout(0.3),
            
            GRU(64, return_sequences=False,
                kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu', kernel_initializer='he_normal'),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        return model
    
    def train(self, X, y, model_type='lstm', test_size=0.2, validation_size=0.2):
        """Train the demand prediction model with comprehensive evaluation"""
        print(f"\n🎯 Training {model_type.upper()} model...")
        self.model_type = model_type
        
        # Scale features
        print("   - Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size/(1-test_size), 
            random_state=42, stratify=y_temp
        )
        
        # Create sequences
        print("   - Creating sequences...")
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, self.sequence_length)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val, self.sequence_length)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test, self.sequence_length)
        
        print(f"   - Training sequences: {X_train_seq.shape}")
        print(f"   - Validation sequences: {X_val_seq.shape}")
        print(f"   - Test sequences: {X_test_seq.shape}")
        
        # Build model
        input_shape = (self.sequence_length, len(self.feature_columns))
        if model_type == 'lstm':
            self.model = self.build_lstm_model(input_shape)
        else:
            self.model = self.build_gru_model(input_shape)
        
        print("   - Model architecture:")
        self.model.summary()
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train model
        print("   - Starting training...")
        history = self.model.fit(
            X_train_seq, y_train_seq,
            epochs=100,
            batch_size=64,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
            shuffle=True
        )
        
        # Evaluate model
        print("\n📈 Model Evaluation:")
        train_metrics = self.model.evaluate(X_train_seq, y_train_seq, verbose=0)
        val_metrics = self.model.evaluate(X_val_seq, y_val_seq, verbose=0)
        test_metrics = self.model.evaluate(X_test_seq, y_test_seq, verbose=0)
        
        print(f"   - Training Accuracy: {train_metrics[1]:.3f}")
        print(f"   - Validation Accuracy: {val_metrics[1]:.3f}")
        print(f"   - Test Accuracy: {test_metrics[1]:.3f}")
        print(f"   - Test Precision: {test_metrics[2]:.3f}")
        print(f"   - Test Recall: {test_metrics[3]:.3f}")
        print(f"   - Test AUC: {test_metrics[4]:.3f}")
        
        # Additional metrics
        y_pred_proba = self.model.predict(X_test_seq)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        print(f"\n📊 Classification Report:")
        print(classification_report(y_test[self.sequence_length:], y_pred))
        
        return history
    
    def predict_demand(self, input_data):
        """Predict demand/cancellation probability for new booking data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Add engineered features
        input_df['total_guests'] = input_df['adults'] + input_df.get('children', 0) + input_df.get('babies', 0)
        input_df['total_nights'] = input_df['stays_in_weekend_nights'] + input_df['stays_in_week_nights']
        input_df['is_weekend_stay'] = (input_df['stays_in_weekend_nights'] > 0).astype(int)
        input_df['has_special_requests'] = (input_df.get('total_of_special_requests', 0) > 0).astype(int)
        input_df['has_parking'] = (input_df.get('required_car_parking_spaces', 0) > 0).astype(int)
        
        # Add temporal features if not provided
        if 'arrival_date' in input_df.columns:
            arrival_date = pd.to_datetime(input_df['arrival_date'].iloc[0])
            input_df['arrival_month'] = arrival_date.month
            input_df['arrival_day_of_week'] = arrival_date.dayofweek
            input_df['arrival_week_of_year'] = arrival_date.isocalendar().week
            input_df['arrival_quarter'] = arrival_date.quarter
            input_df['is_peak_season'] = int(arrival_date.month in [7, 8, 12])
        
        # Season classification
        if 'arrival_month' in input_df.columns:
            month = input_df['arrival_month'].iloc[0]
            if month in [12, 1, 2]:
                input_df['season'] = 'Winter'
            elif month in [3, 4, 5]:
                input_df['season'] = 'Spring'
            elif month in [6, 7, 8]:
                input_df['season'] = 'Summer'
            else:
                input_df['season'] = 'Fall'
        
        # Lead time category
        if 'lead_time' in input_df.columns:
            lead_time = input_df['lead_time'].iloc[0]
            if lead_time <= 7:
                input_df['booking_lead_time_category'] = 'Last Minute'
            elif lead_time <= 30:
                input_df['booking_lead_time_category'] = 'Short'
            elif lead_time <= 90:
                input_df['booking_lead_time_category'] = 'Medium'
            elif lead_time <= 365:
                input_df['booking_lead_time_category'] = 'Long'
            else:
                input_df['booking_lead_time_category'] = 'Very Long'
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col + '_encoded'] = encoder.transform(input_df[col])
                except ValueError:
                    # Handle unseen categories
                    input_df[col + '_encoded'] = 0  # Default to first category
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Select and scale features
        X_input = input_df[self.feature_columns]
        X_scaled = self.scaler.transform(X_input)
        
        # Create sequence (repeating the single data point)
        X_sequence = np.array([X_scaled] * self.sequence_length).reshape(1, self.sequence_length, len(self.feature_columns))
        
        # Make prediction
        prediction = self.model.predict(X_sequence)[0][0]
        
        # Convert cancellation probability to demand confidence
        # Lower cancellation probability = higher demand confidence
        demand_confidence = 1 - prediction
        
        return {
            'demand_confidence': float(demand_confidence),
            'cancellation_probability': float(prediction),
            'demand_level': 'High' if demand_confidence > 0.7 else 'Medium' if demand_confidence > 0.4 else 'Low',
            'features_used': len(self.feature_columns),
            'model_type': self.model_type
        }
    
    def save_model(self, filepath):
        """Save trained model and preprocessing objects"""
        self.model.save(filepath + '_model.h5')
        joblib.dump(self.scaler, filepath + '_scaler.pkl')
        joblib.dump(self.label_encoders, filepath + '_encoders.pkl')
        joblib.dump(self.feature_columns, filepath + '_features.pkl')
        joblib.dump(self.sequence_length, filepath + '_params.pkl')
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model and preprocessing objects"""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath + '_model.h5')
        self.scaler = joblib.load(filepath + '_scaler.pkl')
        self.label_encoders = joblib.load(filepath + '_encoders.pkl')
        self.feature_columns = joblib.load(filepath + '_features.pkl')
        self.sequence_length = joblib.load(filepath + '_params.pkl')
        print(f"✅ Model loaded from {filepath}")

class PricingOptimizer:
    def __init__(self):
        self.base_price = 100
        self.price_elasticity = -1.8  # Price elasticity for hotel rooms
        self.min_price = 50
        self.max_price = 400
        
    def calculate_optimal_price(self, demand_confidence, competition_price, season_factor=1.0, 
                              hotel_type='Resort', room_type='Standard'):
        """Calculate optimal price using advanced pricing model"""
        
        # Base price adjustment based on hotel and room type
        base_price = self.base_price
        if hotel_type == 'Resort':
            base_price *= 1.2  # Resort hotels can charge 20% more
        elif hotel_type == 'City':
            base_price *= 1.0  # Standard pricing for city hotels
        
        # Room type multipliers
        room_multipliers = {
            'Standard': 1.0,
            'Deluxe': 1.3,
            'Suite': 1.7,
            'Presidential': 2.5
        }
        room_multiplier = room_multipliers.get(room_type, 1.0)
        base_price *= room_multiplier
        
        # Demand-based pricing with exponential growth for high demand
        if demand_confidence > 0.8:
            # Very high demand - premium pricing
            demand_multiplier = 1.5 + (demand_confidence - 0.8) * 2.5
        elif demand_confidence > 0.5:
            # Medium to high demand - linear scaling
            demand_multiplier = 1.0 + (demand_confidence - 0.5) * 1.0
        else:
            # Low demand - discounted pricing
            demand_multiplier = 0.7 + demand_confidence * 0.6
        
        # Competition-aware pricing with market positioning
        competition_ratio = competition_price / base_price
        
        if competition_ratio < 0.8:
            # We're more expensive - justify with premium positioning
            competition_adjustment = 1.15
        elif competition_ratio > 1.2:
            # We're cheaper - consider value positioning or match
            competition_adjustment = 0.95
        else:
            # Similar pricing - slight premium for better features
            competition_adjustment = 1.0 + (competition_ratio - 1.0) * 0.1
        
        # Seasonality with smooth transitions
        season_adjustment = 0.8 + (season_factor * 0.7)
        
        # Calculate optimal price
        optimal_price = (base_price * demand_multiplier * 
                       competition_adjustment * season_adjustment)
        
        # Apply psychological pricing (ending with 9, 99, or 95)
        optimal_price = self.apply_psychological_pricing(optimal_price)
        
        # Apply constraints
        optimal_price = max(self.min_price, min(self.max_price, optimal_price))
        
        return {
            'optimal_price': round(optimal_price, 2),
            'base_price': round(base_price, 2),
            'demand_multiplier': round(demand_multiplier, 2),
            'competition_adjustment': round(competition_adjustment, 2),
            'season_adjustment': round(season_adjustment, 2),
            'room_multiplier': room_multiplier,
            'pricing_strategy': self.get_pricing_strategy(demand_confidence, competition_ratio)
        }
    
    def apply_psychological_pricing(self, price):
        """Apply psychological pricing techniques"""
        if price >= 300:
            return round(price - 1) + 0.99  # $299.99 instead of $300
        elif price >= 100:
            return round(price - 1) + 0.99  # $199.99 instead of $200
        elif price >= 50:
            return round(price - 0.05, 2)  # $99.95 instead of $100
        else:
            return round(price, 2)
    
    def get_pricing_strategy(self, demand_confidence, competition_ratio):
        """Determine pricing strategy based on market conditions"""
        if demand_confidence > 0.8 and competition_ratio < 1.0:
            return "Premium Positioning"
        elif demand_confidence > 0.6 and competition_ratio <= 1.2:
            return "Market Leadership"
        elif demand_confidence > 0.4:
            return "Competitive Matching"
        else:
            return "Value Positioning"
    
    def estimate_occupancy(self, price, demand_confidence, base_occupancy=0.7):
        """Estimate occupancy rate based on price elasticity and demand"""
        price_ratio = price / self.base_price
        price_effect = self.price_elasticity * (price_ratio - 1)
        
        # Demand effect - higher demand increases base occupancy
        demand_effect = (demand_confidence - 0.5) * 0.4  # ±20% effect
        
        occupancy = base_occupancy * (1 + price_effect + demand_effect)
        
        # Ensure realistic bounds
        return max(0.1, min(0.95, occupancy))
    
    def calculate_revenue(self, price, occupancy, total_rooms=100):
        """Calculate expected revenue"""
        return price * occupancy * total_rooms

# Example usage and training script
def main():
    print("=" * 70)
    print("🏨 SMARTSTAY HOTEL PRICING OPTIMIZATION SYSTEM")
    print("=" * 70)
    
    # Initialize the predictor and optimizer
    predictor = HotelDemandPredictor()
    optimizer = PricingOptimizer()
    
    try:
        # Load and preprocess data
        X, y, df = predictor.load_and_preprocess_data('hotel_bookings.csv')
        
        if X is not None:
            # Train the model (commented out for demo - would take time)
            # print("\n" + "="*50)
            # print("TRAINING PHASE")
            # print("="*50)
            # history = predictor.train(X, y, model_type='lstm')
            # predictor.save_model('smartstay_model')
            
            # For demo purposes, we'll create a sample prediction
            print("\n" + "="*50)
            print("DEMO PREDICTION")
            print("="*50)
            
            sample_booking = {
                'hotel': 'Resort Hotel',
                'lead_time': 45,
                'arrival_date': '2024-07-15',
                'arrival_month': 7,
                'stays_in_weekend_nights': 2,
                'stays_in_week_nights': 3,
                'adults': 2,
                'children': 0,
                'babies': 0,
                'is_repeated_guest': 0,
                'previous_cancellations': 0,
                'previous_bookings_not_canceled': 0,
                'booking_changes': 0,
                'days_in_waiting_list': 0,
                'adr': 150,
                'required_car_parking_spaces': 1,
                'total_of_special_requests': 2,
                'meal': 'BB',
                'country': 'PRT',
                'market_segment': 'Online TA',
                'distribution_channel': 'TA/TO',
                'deposit_type': 'No Deposit',
                'customer_type': 'Transient',
                'reserved_room_type': 'A',
                'assigned_room_type': 'A'
            }
            
            # Simulate prediction (in real scenario, this would use the trained model)
            demand_prediction = {
                'demand_confidence': 0.78,
                'cancellation_probability': 0.22,
                'demand_level': 'High',
                'features_used': 42,
                'model_type': 'LSTM'
            }
            
            price_optimization = optimizer.calculate_optimal_price(
                demand_prediction['demand_confidence'],
                competition_price=180,
                season_factor=1.2,
                hotel_type='Resort',
                room_type='Deluxe'
            )
            
            occupancy_estimate = optimizer.estimate_occupancy(
                price_optimization['optimal_price'],
                demand_prediction['demand_confidence']
            )
            
            weekly_revenue = optimizer.calculate_revenue(
                price_optimization['optimal_price'],
                occupancy_estimate,
                total_rooms=100
            ) * 7
            
            print(f"📊 Prediction Results:")
            print(f"   - Demand Confidence: {demand_prediction['demand_confidence']:.1%}")
            print(f"   - Demand Level: {demand_prediction['demand_level']}")
            print(f"   - Cancellation Probability: {demand_prediction['cancellation_probability']:.1%}")
            print(f"\n💰 Pricing Optimization:")
            print(f"   - Optimal Price: ${price_optimization['optimal_price']}")
            print(f"   - Base Price: ${price_optimization['base_price']}")
            print(f"   - Pricing Strategy: {price_optimization['pricing_strategy']}")
            print(f"   - Demand Multiplier: {price_optimization['demand_multiplier']}x")
            print(f"   - Competition Adjustment: {price_optimization['competition_adjustment']}x")
            print(f"\n📈 Business Impact:")
            print(f"   - Expected Occupancy: {occupancy_estimate:.1%}")
            print(f"   - Weekly Revenue: ${weekly_revenue:,.0f}")
            print(f"   - Revenue per Available Room: ${price_optimization['optimal_price'] * occupancy_estimate:.2f}")
            
            # Compare with fixed pricing
            fixed_price = 150
            fixed_occupancy = optimizer.estimate_occupancy(fixed_price, demand_prediction['demand_confidence'])
            fixed_revenue = optimizer.calculate_revenue(fixed_price, fixed_occupancy, 100) * 7
            
            revenue_increase = ((weekly_revenue - fixed_revenue) / fixed_revenue) * 100
            
            print(f"\n📊 Vs. Fixed Pricing ($150):")
            print(f"   - Revenue Increase: {revenue_increase:+.1f}%")
            print(f"   - Absolute Increase: ${weekly_revenue - fixed_revenue:,.0f}")
            
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        print("💡 Please ensure the dataset file 'hotel_bookings.csv' is available.")
    
    print("\n" + "="*70)
    print("✅ Demo completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()
