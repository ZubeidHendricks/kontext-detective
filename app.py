"""
Kontext Detective - AI Content Detection App
Entry for Black Forest Labs FLUX.1 Kontext [dev] Hackathon
"""

import streamlit as st
import os
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Import our detection modules (will create these next)
try:
    from detection.detector import KontextDetector
    from detection.kontext_client import KontextClient
    from detection.analyzer import ImageAnalyzer
except ImportError:
    st.error("Detection modules not found. Please ensure all files are in place.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Kontext Detective 🔍",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .ai-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    .real-score { color: #28a745; }
    .ai-score-high { color: #dc3545; }
    .ai-score-medium { color: #ffc107; }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Kontext Detective</h1>
        <p>AI-Generated Content Detection using FLUX.1 Kontext [dev]</p>
        <p><em>Black Forest Labs Hackathon Entry</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "FAL API Key", 
            type="password",
            help="Enter your FAL API key to use FLUX.1 Kontext [dev]"
        )
        
        # Detection settings
        st.header("🔧 Detection Settings")
        
        detection_sensitivity = st.slider(
            "Detection Sensitivity", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.7,
            help="Higher values are more sensitive to AI artifacts"
        )
        
        num_edit_prompts = st.slider(
            "Number of Edit Tests", 
            min_value=1, 
            max_value=10, 
            value=5,
            help="More tests = more accurate but slower"
        )
        
        st.header("📊 About")
        st.markdown("""
        **How it works:**
        1. Upload an image
        2. Apply FLUX.1 Kontext edits
        3. Analyze reconstruction patterns
        4. Generate AI likelihood score
        
        **Categories:**
        - 🏆 Best Overall
        - 💻 Best Local Use Case
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload an image to test for AI generation"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.info(f"""
            **Image Info:**
            - Size: {image.size[0]} x {image.size[1]} pixels
            - Mode: {image.mode}
            - Format: {uploaded_file.type}
            """)
    
    with col2:
        st.header("🔍 Detection Results")
        
        if uploaded_file is not None and api_key:
            # Analysis button
            if st.button("🚀 Analyze Image", type="primary"):
                analyze_image(image, api_key, detection_sensitivity, num_edit_prompts)
        elif uploaded_file is not None:
            st.warning("⚠️ Please enter your FAL API key in the sidebar to begin analysis.")
        else:
            st.info("👆 Upload an image to start detection analysis.")
    
    # Results section
    if 'analysis_results' in st.session_state:
        display_results(st.session_state.analysis_results)

def analyze_image(image, api_key, sensitivity, num_prompts):
    """Analyze uploaded image for AI detection"""
    
    with st.spinner("🔄 Analyzing image with FLUX.1 Kontext [dev]..."):
        try:
            # Initialize detector
            detector = KontextDetector(api_key=api_key)
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Test API connection
            status_text.text("Testing API connection...")
            progress_bar.progress(10)
            
            if not detector.test_connection():
                st.error("❌ Failed to connect to FLUX.1 Kontext API. Please check your API key.")
                return
            
            # Step 2: Run detection analysis
            status_text.text("Running detection analysis...")
            progress_bar.progress(30)
            
            results = detector.detect(
                image, 
                sensitivity=sensitivity,
                num_prompts=num_prompts,
                progress_callback=lambda p: progress_bar.progress(30 + int(p * 0.6))
            )
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Store results in session state
            st.session_state.analysis_results = results
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)

def display_results(results):
    """Display analysis results"""
    
    st.header("📊 Analysis Results")
    
    # Overall AI likelihood score
    ai_score = results.get('ai_likelihood', 0.5)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score_class = "real-score" if ai_score < 0.3 else "ai-score-high" if ai_score > 0.7 else "ai-score-medium"
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 AI Likelihood</h3>
            <div class="ai-score {score_class}">{ai_score:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        confidence = results.get('confidence', 0.5)
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎲 Confidence</h3>
            <div class="ai-score">{confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        processing_time = results.get('processing_time', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>⏱️ Processing Time</h3>
            <div class="ai-score">{processing_time:.1f}s</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Interpretation
    if ai_score < 0.3:
        st.success("✅ **Likely REAL**: This image shows patterns consistent with authentic photography.")
    elif ai_score > 0.7:
        st.error("❌ **Likely AI-GENERATED**: This image shows strong patterns consistent with AI generation.")
    else:
        st.warning("⚠️ **UNCERTAIN**: Mixed signals detected. Manual review recommended.")
    
    # Detailed metrics
    st.subheader("📈 Detailed Analysis")
    
    tabs = st.tabs(["Edit Sensitivity", "Reconstruction Artifacts", "Consistency Analysis", "Visual Comparison"])
    
    with tabs[0]:
        display_sensitivity_analysis(results.get('sensitivity', {}))
    
    with tabs[1]:
        display_artifact_analysis(results.get('artifacts', {}))
    
    with tabs[2]:
        display_consistency_analysis(results.get('consistency', 0.5))
    
    with tabs[3]:
        display_visual_comparison(results.get('visual_comparison', {}))

def display_sensitivity_analysis(sensitivity_data):
    """Display edit sensitivity analysis"""
    
    if not sensitivity_data:
        st.info("No sensitivity data available.")
        return
    
    # Create sensitivity chart
    prompts = list(sensitivity_data.keys())
    scores = list(sensitivity_data.values())
    
    fig = px.bar(
        x=prompts,
        y=scores,
        title="Edit Sensitivity by Prompt",
        labels={'x': 'Edit Prompt', 'y': 'Sensitivity Score'},
        color=scores,
        color_continuous_scale='RdYlGn_r'
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Interpretation**: Higher sensitivity scores suggest the image changes more dramatically 
    when edited, which can indicate AI generation.
    """)

def display_artifact_analysis(artifacts_data):
    """Display reconstruction artifact analysis"""
    
    if not artifacts_data:
        st.info("No artifact data available.")
        return
    
    # Display artifact metrics for each prompt
    for prompt, artifacts in artifacts_data.items():
        st.subheader(f"Prompt: {prompt}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Difference", f"{artifacts.get('mean_difference', 0):.2f}")
        
        with col2:
            st.metric("Max Difference", f"{artifacts.get('max_difference', 0):.2f}")
        
        with col3:
            st.metric("Artifact Density", f"{artifacts.get('artifact_density', 0):.3f}")
        
        with col4:
            st.metric("High Freq Artifacts", f"{artifacts.get('high_freq_artifacts', 0):.2f}")

def display_consistency_analysis(consistency_score):
    """Display consistency analysis"""
    
    st.metric("Consistency Score", f"{consistency_score:.3f}")
    
    # Consistency gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = consistency_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Edit Consistency"},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "gray"},
                {'range': [0.7, 1], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Interpretation**: Lower consistency scores may indicate AI generation, 
    as AI images often respond inconsistently to different editing prompts.
    """)

def display_visual_comparison(visual_data):
    """Display visual comparison of original vs edited images"""
    
    if not visual_data:
        st.info("No visual comparison data available.")
        return
    
    st.info("Visual comparison charts would be displayed here showing original vs edited images and difference maps.")

# Demo section
def show_demo():
    """Show demo with sample results"""
    
    st.header("🎬 Demo Mode")
    st.info("This demo shows sample results. Upload your own image for real analysis.")
    
    # Sample results
    sample_results = {
        'ai_likelihood': 0.85,
        'confidence': 0.92,
        'processing_time': 15.3,
        'sensitivity': {
            'enhance quality': 0.45,
            'improve lighting': 0.62,
            'sharpen details': 0.38,
            'make vibrant': 0.71,
            'reduce noise': 0.29
        },
        'artifacts': {},
        'consistency': 0.34,
        'visual_comparison': {}
    }
    
    st.session_state.analysis_results = sample_results
    display_results(sample_results)

if __name__ == "__main__":
    # Add demo mode toggle
    if st.sidebar.button("🎬 Show Demo"):
        show_demo()
    
    main()