import streamlit as st
import pandas as pd
import joblib
import os

st.title("Delinquency Risk Prediction System")

# -------------------------------
# Upload Borrower Dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload Borrower CSV", type=["csv"])
if not uploaded_file:
   # st.info("Please upload a borrower data CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("Uploaded Data", df)

# -------------------------------
# Level 1: Feature Risk Prediction
# -------------------------------
st.header("Level 1: Feature Risk Prediction")
level1_dir = "models/level1"
level1_risks = []

for file in os.listdir(level1_dir):
    if file.endswith("_model.pkl"):
        factor = file.replace("_model.pkl", "")
        #st.write(factor)
        if factor not in df.columns:
            st.warning(f"Skipping {factor} — column not found in input.")
            continue

        model_data = joblib.load(os.path.join(level1_dir, file))
      
        model = model_data["model"]
      
        enc_X = model_data["encoder_X"]
        
        enc_y = model_data["encoder_y"]
        

        try:
            X = df[[factor]]
            if enc_X:
                X_enc = enc_X.transform(X)
            else:
                X_enc = X.values

            preds = model.predict(X_enc.reshape(-1, 1))
            
            labels = enc_y.inverse_transform(preds)
            
            df[factor + "Risk"] = labels
            
            level1_risks.append(factor + "Risk")
            
            
        except Exception as e:
            
            st.error(f"Error processing {factor}: {e}")

if level1_risks:
    st.success("Level 1 Predictions Done")
    
    st.dataframe(df[level1_risks])
else:
    st.error("No Level 1 predictions made.")
    st.stop()

# -------------------------------
# Level 2: Group Risk Prediction
# -------------------------------
st.header("Level 2: Group Risk Prediction")
level2_dir = "models/level2"
level2_risks = []

for file in os.listdir(level2_dir):
    if file.endswith("_model.pkl"):
        
        group = file.replace("_model.pkl", "")
        
        model_data = joblib.load(os.path.join(level2_dir, file))
        
        model = model_data["model"]
        
        encoders = model_data["encoders"]
        
        #st.write(encoders)
        
        input_cols = []
        
        output_col = None
        
        for col in encoders:
            
            if col not in df.columns:
                
                output_col = col
                
                break

        if not output_col:
            st.warning(f"Could not identify output column for {group}")
            continue

        for col, enc in encoders.items():
            if col == output_col:
                continue
            if col in df.columns:
                try:
                    df[col + "Encoded"] = enc.transform(df[col])
                    
                    input_cols.append(col + "Encoded")
                except Exception as e:
                    st.warning(f"Encoding error for {col}: {e}")
            else:
                st.warning(f"Missing input column for {group}: {col}")

        if input_cols:
            
            X = df[input_cols]
            
            preds = model.predict(X)
            
            decoded = encoders[output_col].inverse_transform(preds)
            
            df[output_col] = decoded
            
            level2_risks.append(output_col)

if level2_risks:
    st.success("Level 2 Predictions Done")
    st.dataframe(df[level2_risks])
else:
    st.warning("No Level 2 predictions made.")
    st.stop()

# -------------------------------
# Level 3: Final Risk Prediction
# -------------------------------
st.header("Level 3: Final Risk Prediction")

level3_model_path = "models/level3/level3_model.pkl"

# Manual mapping for fallback decoding
label_map = {
    0: "Very Low",
    1: "Low",
    2: "Low Moderate",
    3: "Moderate",
    4: "High Moderate",
    5: "High",
    6: "Very High"
}

if os.path.exists(level3_model_path):
    model_data = joblib.load(level3_model_path)
    
    model = model_data["model"]
    
    encoders = model_data["encoders"]
    
    required_features = model_data["feature_names"]
    #st.write(required_features)

    input_cols = []
    
    for col in required_features:
        if col in df.columns:
            input_cols.append(col)
       
        elif col.replace("Encoded", "") in df.columns:
            
            raw_col = col.replace("Encoded", "")
            
            df[col] = encoders[raw_col].transform(df[raw_col])
            
            input_cols.append(col)
            
   # st.write(input_cols)
    if set(required_features) == set(input_cols):
        
        X_final = df[input_cols]
        
        preds = model.predict(X_final)

        try:
            final_encoder = encoders.get("Final Risk")
            
            if final_encoder:
                df["FinalRiskLabel"] = final_encoder.inverse_transform(preds)
            else:
                df["FinalRiskLabel"] = [label_map.get(p, f"Unknown({p})") for p in preds]
                
        except Exception as e:
            st.warning(f"Decoder failed, using fallback mapping: {e}")
            
            df["FinalRiskLabel"] = [label_map.get(p, f"Unknown({p})") for p in preds]

        st.success("Final Risk Prediction Done")
        st.dataframe(df[["FinalRiskLabel"]])

    else:
        
        missing = set(required_features) - set(input_cols)
        
        st.error(f"Missing required features for Level 3 prediction: {missing}")
else:
    st.error("Level 3 model file not found.")
    
    

# -------------------------------
# Download Results
# -------------------------------
st.markdown("---")
st.download_button(
    label="📥 Download All Predictions as CSV",
    data=df.to_csv(index=False),
    file_name="full_risk_predictions.csv",
    mime="text/csv"
)

