import gradio as gr
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("penguin_model_with_island.joblib")


# Prediction function
def predict_species(
    island, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g, sex
):
    input_data = pd.DataFrame(
        {
            "island": [island],
            "culmen_length_mm": [culmen_length_mm],
            "culmen_depth_mm": [culmen_depth_mm],
            "flipper_length_mm": [flipper_length_mm],
            "body_mass_g": [body_mass_g],
            "sex": [sex],
        }
    )
    prediction = model.predict(input_data)
    return prediction[0]


# Define the Gradio interface with custom styling
interface = gr.Interface(
    fn=predict_species,
    inputs=[
        gr.Dropdown(["Biscoe", "Dream", "Torgersen"], label="Island"),
        gr.Number(label="Culmen Length (mm)", value=0),
        gr.Number(label="Culmen Depth (mm)", value=0),
        gr.Number(label="Flipper Length (mm)", value=0),
        gr.Number(label="Body Mass (g)", value=0),
        gr.Radio(["male", "female"], label="Sex"),
    ],
    outputs=gr.Textbox(label="Predicted Species"),
    title="Penguin Species Predictor",
    description="Predict the species of a penguin based on input features.",
    theme="default",  # You can change this to 'default', 'compact', or 'huggingface'
    css="""
    .gradio-container {
        background-color: #9; /* Change background color */
        border-radius: 15px; /* Rounded corners */
        padding: 20px; /* Padding inside the container */
    }
    h1, h2, h3 {
        color: #007BFF; /* Header color */
    }
    .gr-button {
        background-color: #28a745; /* Button color */
        color: white; /* Button text color */
    }
    .gr-button:hover {
        background-color: #918038; /* Button hover color */
    }
    """,
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
