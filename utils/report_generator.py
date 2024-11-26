from fpdf import FPDF
import datetime

def generate_report(results_df, prompt_type, accuracy):
    """
    Generate PDF report with analysis results
    """
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Sentiment Analysis Report', ln=True, align='C')
    
    # Date and Configuration
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True)
    pdf.cell(0, 10, f'Model: LLaMA 3.1 8B', ln=True)
    pdf.cell(0, 10, f'Prompt Type: {prompt_type}', ln=True)
    pdf.cell(0, 10, f'Accuracy: {accuracy:.2f}%', ln=True)
    
    # Results Table
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Detailed Results', ln=True)
    
    # Table headers
    headers = ['Text', 'True Label', 'Predicted Label', 'Correct']
    col_widths = [100, 30, 30, 30]
    
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, 1)
    pdf.ln()
    
    # Table content
    pdf.set_font('Arial', '', 10)
    for _, row in results_df.iterrows():
        pdf.cell(col_widths[0], 10, str(row['text'])[:50] + '...', 1)
        pdf.cell(col_widths[1], 10, row['true_label'], 1)
        pdf.cell(col_widths[2], 10, row['predicted_label'], 1)
        pdf.cell(col_widths[3], 10, str(row['correct']), 1)
        pdf.ln()
    
    # Save report
    report_path = "sentiment_analysis_report.pdf"
    pdf.output(report_path)
    return report_path
