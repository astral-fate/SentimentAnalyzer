import streamlit as st
import pandas as pd
from utils.data_loader import load_imdb_dataset
from utils.groq_client import GroqAnalyzer
from utils.prompt_templates import zero_shot_prompt, few_shot_prompt
from utils.report_generator import generate_report

st.set_page_config(page_title="Sentiment Analysis with LLaMA 3.1", layout="wide")

def main():
    st.title("Sentiment Analysis System using LLaMA 3.1")
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    prompt_type = st.sidebar.selectbox("Select Prompt Type", ["Zero-shot", "Few-shot"])
    
    if not api_key:
        st.warning("Please enter your Groq API Key to proceed.")
        return
    
    # Initialize analyzer
    analyzer = GroqAnalyzer(api_key)
    
    # Load dataset
    if st.button("Load Dataset & Run Analysis"):
        with st.spinner("Loading IMDB dataset..."):
            dataset = load_imdb_dataset()
            
        # Select first 100 examples from test set
        test_samples = dataset['test'].select(range(100))
        
        results = []
        accuracies = {'Zero-shot': 0, 'Few-shot': 0}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, sample in enumerate(test_samples):
            progress = (idx + 1) / len(test_samples)
            progress_bar.progress(progress)
            status_text.text(f"Processing example {idx + 1}/{len(test_samples)}")
            
            # Get prediction
            text = sample['text']
            true_label = "positive" if sample['label'] == 1 else "negative"
            
            if prompt_type == "Zero-shot":
                prompt = zero_shot_prompt(text)
            else:
                prompt = few_shot_prompt(text)
            
            predicted_label = analyzer.analyze(prompt)
            
            results.append({
                'text': text[:100] + '...',
                'true_label': true_label,
                'predicted_label': predicted_label,
                'correct': true_label == predicted_label
            })
        
        # Calculate accuracy
        df_results = pd.DataFrame(results)
        accuracy = (df_results['correct'].sum() / len(df_results)) * 100
        
        # Display results
        st.subheader("Analysis Results")
        st.write(f"Accuracy ({prompt_type}): {accuracy:.2f}%")
        
        # Display results table
        st.dataframe(df_results)
        
        # Generate PDF report
        if st.button("Generate Report"):
            with st.spinner("Generating PDF report..."):
                report_path = generate_report(df_results, prompt_type, accuracy)
                st.success(f"Report generated successfully!")
                
                with open(report_path, "rb") as f:
                    st.download_button(
                        label="Download Report",
                        data=f,
                        file_name="sentiment_analysis_report.pdf",
                        mime="application/pdf"
                    )

if __name__ == "__main__":
    main()
