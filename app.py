"""
AI-Based Semantic Evaluation of Descriptive Answers - Streamlit Application

This is the main Streamlit application that provides a user-friendly interface
for evaluating student answers using semantic similarity and AI-based scoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from pathlib import Path
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our custom modules
from data_preprocessing import DataPreprocessor
from semantic_evaluator import SemanticEvaluator, AdvancedSemanticEvaluator
from bert_scorer import BERTScorer
from feedback_generator import LocalFeedbackGenerator, OpenAIFeedbackGenerator
from evaluation_metrics import EvaluationMetrics

# Page configuration
st.set_page_config(
    page_title="AI-Based Semantic Evaluation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .feedback-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .warning-message {
        color: #ffc107;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None
if 'scorer' not in st.session_state:
    st.session_state.scorer = None
if 'feedback_generator' not in st.session_state:
    st.session_state.feedback_generator = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI-Based Semantic Evaluation of Descriptive Answers</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("Model Settings")
        semantic_model = st.selectbox(
            "Semantic Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"],
            help="Choose the Sentence-BERT model for semantic similarity"
        )
        
        scoring_model = st.selectbox(
            "Scoring Model",
            ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"],
            help="Choose the BERT model for score prediction"
        )
        
        feedback_type = st.selectbox(
            "Feedback Generator",
            ["Local Analysis", "OpenAI GPT"],
            help="Choose the feedback generation method"
        )
        
        # OpenAI API key input
        if feedback_type == "OpenAI GPT":
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key for GPT-based feedback"
            )
            if openai_key:
                os.environ['OPENAI_API_KEY'] = openai_key
        
        # Initialize models button
        if st.button("🚀 Initialize Models", type="primary"):
            with st.spinner("Initializing models..."):
                try:
                    st.session_state.evaluator = SemanticEvaluator(semantic_model)
                    st.session_state.scorer = BERTScorer(scoring_model)
                    st.session_state.feedback_generator = (
                        OpenAIFeedbackGenerator() if feedback_type == "OpenAI GPT" 
                        else LocalFeedbackGenerator()
                    )
                    st.success("✅ Models initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Error initializing models: {str(e)}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Single Answer Evaluation", 
        "📊 Batch Evaluation", 
        "📈 Model Training", 
        "📋 Dataset Management",
        "📊 Analytics Dashboard"
    ])
    
    with tab1:
        single_answer_evaluation()
    
    with tab2:
        batch_evaluation()
    
    with tab3:
        model_training()
    
    with tab4:
        dataset_management()
    
    with tab5:
        analytics_dashboard()

def single_answer_evaluation():
    """Single answer evaluation interface."""
    st.header("📝 Single Answer Evaluation")
    
    if not st.session_state.evaluator:
        st.warning("⚠️ Please initialize models in the sidebar first.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        
        question = st.text_area(
            "Question",
            placeholder="Enter the question here...",
            height=100
        )
        
        model_answer = st.text_area(
            "Model Answer (Teacher's Answer)",
            placeholder="Enter the model/teacher answer here...",
            height=150
        )
        
        student_answer = st.text_area(
            "Student Answer",
            placeholder="Enter the student's answer here...",
            height=150
        )
        
        if st.button("🔍 Evaluate Answer", type="primary"):
            if not all([question, model_answer, student_answer]):
                st.error("❌ Please fill in all fields.")
                return
            
            with st.spinner("Evaluating answer..."):
                try:
                    # Compute semantic similarity
                    similarity = st.session_state.evaluator.compute_similarity(
                        model_answer, student_answer
                    )
                    
                    # Generate feedback
                    feedback = st.session_state.feedback_generator.generate_feedback(
                        question, model_answer, student_answer, similarity * 10
                    )
                    
                    # Store results
                    st.session_state.evaluation_results = {
                        'question': question,
                        'model_answer': model_answer,
                        'student_answer': student_answer,
                        'similarity': similarity,
                        'feedback': feedback
                    }
                    
                    st.success("✅ Evaluation completed!")
                    
                except Exception as e:
                    st.error(f"❌ Error during evaluation: {str(e)}")
    
    with col2:
        st.subheader("Results")
        
        if st.session_state.evaluation_results:
            results = st.session_state.evaluation_results
            
            # Score display
            score = results['similarity'] * 10
            st.metric("Semantic Similarity Score", f"{score:.1f}/10")
            
            # Similarity interpretation
            interpretation = st.session_state.evaluator.get_similarity_interpretation(
                results['similarity']
            )
            st.info(f"📊 {interpretation}")
            
            # Feedback display
            st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
            st.subheader("📝 Detailed Feedback")
            
            feedback = results['feedback']
            
            st.write("**Overall Assessment:**")
            st.write(feedback['overall_feedback'])
            
            # Component feedback
            components = ['coverage', 'relevance', 'grammar', 'coherence']
            for component in components:
                if component in feedback:
                    st.write(f"**{component.title()}:**")
                    st.write(feedback[component]['feedback'])
                    st.progress(feedback[component]['score'])
            st.markdown('</div>', unsafe_allow_html=True)

def batch_evaluation():
    """Batch evaluation interface."""
    st.header("📊 Batch Evaluation")
    
    if not st.session_state.evaluator:
        st.warning("⚠️ Please initialize models in the sidebar first.")
        return
    
    st.subheader("Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: question, model_answer, student_answer, score"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset loaded successfully! ({len(df)} rows)")
            
            # Display dataset preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            
            # Validate columns
            required_columns = ['question', 'model_answer', 'student_answer']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
                return
            
            if st.button("🚀 Run Batch Evaluation", type="primary"):
                with st.spinner("Running batch evaluation..."):
                    try:
                        # Run evaluation
                        results = st.session_state.evaluator.evaluate_dataset(
                            df['question'].tolist(),
                            df['model_answer'].tolist(),
                            df['student_answer'].tolist()
                        )
                        
                        # Display results
                        st.subheader("Evaluation Results")
                        
                        # Statistics
                        stats = results['statistics']
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Mean Similarity", f"{stats['mean']:.3f}")
                        with col2:
                            st.metric("Std Deviation", f"{stats['std']:.3f}")
                        with col3:
                            st.metric("Min Similarity", f"{stats['min']:.3f}")
                        with col4:
                            st.metric("Max Similarity", f"{stats['max']:.3f}")
                        
                        # Results table
                        results_df = pd.DataFrame(results['detailed_results'])
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="evaluation_results.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during batch evaluation: {str(e)}")
                        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

def model_training():
    """Model training interface."""
    st.header("📈 Model Training")
    
    st.info("🚧 Model training functionality is under development. This would allow you to fine-tune the BERT scoring model on your own dataset.")
    
    # Placeholder for future training functionality
    st.subheader("Training Dataset Requirements")
    st.markdown("""
    To train a custom scoring model, you'll need:
    - A dataset with question-answer-score triplets
    - At least 100 examples for meaningful training
    - Balanced score distribution across the 0-10 range
    - High-quality human-annotated scores
    """)
    
    # Sample training interface (placeholder)
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Training Epochs", 1, 10, 3)
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32])
        learning_rate = st.selectbox("Learning Rate", [1e-5, 2e-5, 5e-5])
    
    with col2:
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
        early_stopping = st.checkbox("Early Stopping", value=True)
        save_best_model = st.checkbox("Save Best Model", value=True)
    
    if st.button("🚀 Start Training", type="primary", disabled=True):
        st.info("Training functionality will be available in a future update.")

def dataset_management():
    """Dataset management interface."""
    st.header("📋 Dataset Management")
    
    st.subheader("Create Sample Dataset")
    
    if st.button("📝 Generate Sample Dataset", type="primary"):
        with st.spinner("Creating sample dataset..."):
            try:
                preprocessor = DataPreprocessor()
                preprocessor.create_sample_dataset()
                st.success("✅ Sample dataset created successfully!")
                
                # Display the created dataset
                df = preprocessor.processed_data
                st.subheader("Sample Dataset")
                st.dataframe(df)
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Sample Dataset",
                    data=csv,
                    file_name="sample_dataset.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Error creating sample dataset: {str(e)}")
    
    st.subheader("Dataset Statistics")
    
    # Placeholder for dataset analysis
    st.info("Upload a dataset to see detailed statistics and analysis.")

def analytics_dashboard():
    """Analytics dashboard."""
    st.header("📊 Analytics Dashboard")
    
    st.info("🚧 Analytics dashboard is under development. This will provide comprehensive insights into model performance and evaluation trends.")
    
    # Placeholder charts
    st.subheader("Performance Metrics")
    
    # Sample data for demonstration
    sample_data = {
        'Metric': ['Pearson Correlation', 'R² Score', 'MSE', 'MAE'],
        'Value': [0.85, 0.72, 1.2, 0.9],
        'Target': [0.8, 0.7, 1.0, 0.8]
    }
    
    df_metrics = pd.DataFrame(sample_data)
    
    # Create a bar chart
    fig = px.bar(
        df_metrics, 
        x='Metric', 
        y='Value',
        title='Model Performance Metrics',
        color='Value',
        color_continuous_scale='RdYlGn'
    )
    
    # Add target lines
    for i, target in enumerate(df_metrics['Target']):
        fig.add_hline(y=target, line_dash="dash", line_color="red", 
                     annotation_text=f"Target: {target}")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample similarity distribution
    st.subheader("Similarity Score Distribution")
    
    # Generate sample data
    np.random.seed(42)
    similarities = np.random.beta(2, 2, 1000)  # Beta distribution for 0-1 range
    
    fig_hist = px.histogram(
        x=similarities,
        nbins=30,
        title="Distribution of Semantic Similarity Scores",
        labels={'x': 'Similarity Score', 'y': 'Frequency'}
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)

def footer():
    """Application footer."""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 AI-Based Semantic Evaluation of Descriptive Answers</p>
        <p>Built with Streamlit, Sentence-BERT, and BERT</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    footer()
