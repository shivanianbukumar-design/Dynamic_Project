import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import os
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------
# Load API Key
# -------------------------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(layout="wide")
st.title("Dynamic Multi-Objective Energy-Aware Optimization")

st.sidebar.header("Experiment Settings")
epochs = st.sidebar.slider("Max Epochs", 5, 20, 10)

# -------------------------------
# Initial Parameters
# -------------------------------

learning_rate = 0.01
batch_size = 64
model_type = "CNN"

accuracy = 60
energy = 0

accuracy_history = []
energy_history = []
decision_history = []

# -------------------------------
# LLM AGENT FUNCTION
# -------------------------------

def llm_controller(acc, energy, lr, batch, model):

    prompt = f"""
    You are an AI optimization agent.

    Current metrics:
    Accuracy: {acc}
    Energy: {energy}
    Learning rate: {lr}
    Batch size: {batch}
    Model type: {model}

    Goal:
    Balance accuracy and energy consumption.

    Suggest one action:
    - increase_lr
    - decrease_lr
    - increase_batch
    - decrease_batch
    - switch_model
    - stop

    Respond with only one action word.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    action = response.choices[0].message.content.strip()
    return action

# -------------------------------
# Simulation Training Step
# -------------------------------

def train_step(model, lr, batch):

    global accuracy, energy

    base_gain = {
        "MLP": 1.5,
        "CNN": 2.5,
        "Transformer": 3.0
    }

    energy_cost = {
        "MLP": 0.002,
        "CNN": 0.004,
        "Transformer": 0.006
    }

    accuracy += base_gain[model] * (lr * 10)
    energy += energy_cost[model] * (batch / 64)

    return accuracy, energy

# -------------------------------
# RUN EXPERIMENT
# -------------------------------

if st.button("Start Dynamic Optimization"):

    progress = st.progress(0)

    # Initialize variables locally
    learning_rate = 0.01
    batch_size = 64
    model_type = "CNN"

    accuracy = 60
    energy = 0

    accuracy_history = []
    energy_history = []
    decision_history = []

    for epoch in range(epochs):

        # Simulate training step
        base_gain = {
            "MLP": 1.5,
            "CNN": 2.5,
            "Transformer": 3.0
        }

        energy_cost = {
            "MLP": 0.002,
            "CNN": 0.004,
            "Transformer": 0.006
        }

        accuracy += base_gain[model_type] * (learning_rate * 10)
        energy += energy_cost[model_type] * (batch_size / 64)

        accuracy_history.append(accuracy)
        energy_history.append(energy)

        score = accuracy - (energy * 100)

        # Call LLM
        action = llm_controller(
            accuracy, energy,
            learning_rate, batch_size,
            model_type
        )

        decision_history.append(action)

        # Apply decision
        if action == "increase_lr":
            learning_rate *= 1.1
        elif action == "decrease_lr":
            learning_rate *= 0.9
        elif action == "increase_batch":
            batch_size += 16
        elif action == "decrease_batch":
            batch_size = max(16, batch_size - 16)
        elif action == "switch_model":
            if model_type == "MLP":
                model_type = "CNN"
            elif model_type == "CNN":
                model_type = "Transformer"
        elif action == "stop":
            break

        st.write(f"Epoch {epoch+1}")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Energy: {energy:.4f}")
        st.write(f"LLM Decision: {action}")
        st.write("---")

        progress.progress((epoch+1)/epochs)
        time.sleep(0.5)

    st.success("Optimization Completed")

    # -------------------------------
    # Visualization
    # -------------------------------

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(accuracy_history)+1)),
        y=accuracy_history,
        mode='lines',
        name='Accuracy'
    ))

    fig.add_trace(go.Scatter(
        x=list(range(1, len(energy_history)+1)),
        y=energy_history,
        mode='lines',
        name='Energy'
    ))

    fig.update_layout(template="plotly_dark",
                      title="Dynamic Training Metrics")

    st.plotly_chart(fig, use_container_width=True)

    # Show decisions
    df = pd.DataFrame({
        "Epoch": list(range(1, len(decision_history)+1)),
        "LLM Decision": decision_history
    })

    st.subheader("LLM Adaptive Decisions")
    st.dataframe(df)