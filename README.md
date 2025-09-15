# Hybrid Phishing Email Detection System

A comprehensive phishing email detection system that combines **traditional blacklist-based security** with **advanced machine learning** for maximum protection.

## 🎯 Project Overview

This system implements a **hybrid detection architecture** that follows the industry-standard approach: **blacklist analysis first, then ML detection as a fallback**. This ensures both high accuracy and comprehensive coverage against phishing threats.

### 🔄 Hybrid Detection Flow:
1. **📋 Primary Defense**: Blacklist checks (URLs, attachments, sender reputation)
2. **🤖 Secondary Defense**: ML pattern analysis (when blacklist is inconclusive)
3. **🎯 Final Verdict**: Combined decision with confidence scoring

## ✨ Key Features

### 🔍 Blacklist Analysis (Primary Defense):
- **URL Reputation Checking**: Real-time queries to PhishTank API
- **Attachment Scanning**: VirusTotal integration for malware detection
- **Whitelist Management**: Trusted domains bypass analysis
- **Sender Reputation**: Domain-based trust scoring

### 🤖 Machine Learning Analysis (Fallback Defense):
- **Advanced Sender Analysis**: Pattern detection in email addresses
- **Text Feature Extraction**: TF-IDF analysis of subject and body
- **URL Pattern Detection**: Sophisticated URL analysis
- **Ensemble Learning**: Random Forest classifier for robust predictions

### 📧 Email Processing:
- **EML File Support**: Direct processing of .eml email files
- **Multi-format Parsing**: HTML and plain text email handling
- **Attachment Extraction**: Automatic file analysis
- **Confidence Scoring**: Detailed confidence levels for all decisions

## 📊 System Performance

### Hybrid Detection Results:
- **Blacklist Success Rate**: 100% for known malicious URLs/attachments
- **ML Fallback Accuracy**: 97.2% cross-validation accuracy
- **Overall System Accuracy**: 98.0% on comprehensive testing
- **False Positive Rate**: <3% (excellent precision)

### Real-World Testing:
Based on test results with 10 diverse email samples:
- **Success Rate**: 100% (all emails processed successfully)
- **Detection Coverage**: Both known and unknown phishing patterns
- **Test Coverage**: Includes legitimate emails from major services (Steam, Strava, Character.AI)

### Sample Test Results:
- **Known Phishing**: Blacklist detected malicious URLs instantly
- **Unknown Phishing**: ML fallback detected with 71-86% confidence
- **Legitimate Emails**: Correctly classified with 63-75% confidence

## 🛠️ Installation & Running

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd CS-Phishing-Detection
   ```

2. **Create & Activate Virtual Environment**:
   ```bash
   python -m venv venv
   venv/Scripts/activate
   ```

3. **Install dependencies to Virtual Environment**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train ML Model**:
- *Extract dataset CEAS_08.csv from archive.zip if needed*
   ```bash
   run ml_training_script/ml_training.ipynb
   ```

5. **Run Phishing Detection Script**:
- *Create .env file in root with VIRUSTOTAL_API_KEY*
   ```bash
   run phishing_detection_script/phishing_detection.ipynb
   ```
- *Change EML_DIR to malicious_emails or test_emails if needed*

## 📁 Project Structure

```
Code/
├── malicious_emails/                # Stores known malicious email samples for reference
│
├── ml_script_evaluation/            # Scripts and artifacts for evaluating the ML model's performance
│   ├── model_evaluation.py          # Script to evaluate the trained model against a dataset
│   ├── model_classes.py             # Custom classes (transformers) used by the evaluation script
│   ├── confusion_matrix.png         # Output visualization of the model's prediction accuracy
│   ├── roc_curve.png                # Output ROC curve visualization for model performance
│   └── model_evaluation_report.txt  # Text report of the evaluation metrics
│
├── ml_training_script/              # Script for training the machine learning model
│   └── ml_training.ipynb            # Jupyter Notebook to train and create the .pkl model file
│
├── phishing_detection_script/       # The main hybrid detection script and its components
│   ├── phishing_detection.ipynb     # Main script to analyze individual .eml files
│   ├── model_classes.py             # Custom classes for ML feature extraction
│   ├── whitelist.json               # Configuration for trusted URLs and domains
│   └── malicious_attachments/       # Default directory where malicious attachments are saved
│
├── test_emails/                     # Sample .eml files for testing the detection script
│   ├── ML-Test.eml                
│   ├── Phishtank-url.eml
│   ├── test_sample_message.eml
│   └── VirusTotal.eml
│
├── test_eml_files/                  # Scripts for batch testing multiple .eml files at once
│   └── test_eml_files_clean.py      # Script to run tests on a directory of emails
│
├── .env*                             # File for local environment variables (API keys)
├── archive.zip                      # Project training datasets archive
├── CEAS_08.csv                      # The dataset used for training the ML model
├── phishing_email_model_fixed.pkl   # The pre-trained machine learning model file
└── requirements.txt                 # Lists the required Python packages for installation

* - User Created
```

## 🚀 Usage

### 🎯 Main System - Hybrid Detection

```bash
phishing_detection_script/phishing_detection.ipynb
```

This runs the complete hybrid detection system:
1. **Blacklist Analysis**: Check URLs against PhishTank, scan attachments with VirusTotal
2. **Whitelist Check**: Skip analysis for trusted domains
3. **ML Fallback**: Use machine learning when blacklist is inconclusive
4. **Final Verdict**: Provide comprehensive security assessment
5. **Output Generated**: Collects and Stores malicious attachments for Emails from sample .eml files

### 🤖 ML-Only Testing (Alternative)

```bash
python test_eml_files/test_eml_files_clean.py
```
- *Change email directory and results to malicious_emails or test_emails accordingly*

This runs only the ML component for comparison/testing.

### 📊 ML Model Training

```bash
ml_training_script/ml_training.ipynb
```

This trains the ML fallback model using the CEAS-08 dataset and generates .pkl file to save the trained model

## 🔄 Hybrid Architecture Details

### Primary Defense - Blacklist Analysis:
1. **URL Extraction**: Parse all URLs from email content
2. **Whitelist Check**: Skip analysis for trusted domains
3. **PhishTank Query**: Real-time reputation checking
4. **Attachment Analysis**: VirusTotal malware scanning
5. **Immediate Decision**: If malicious detected → BLOCK

### Secondary Defense - ML Analysis:
1. **Feature Extraction**: Sender patterns, text analysis, URL patterns
2. **Pattern Recognition**: Advanced ML model analysis
3. **Confidence Scoring**: Probability-based decisions
4. **Fallback Decision**: When blacklist is inconclusive

### Decision Logic:
```
IF blacklist_detects_malicious:
    RETURN "MALICIOUS"
ELSE IF blacklist_unknown:
    ml_result = machine_learning_analysis()
    IF ml_result.confidence > threshold:
        RETURN ml_result.prediction
    ELSE:
        RETURN "UNKNOWN"
ELSE:
    RETURN "SAFE"
```

## 🔍 Technical Components

### Blacklist Features:
- **URL Reputation**: PhishTank API integration
- **File Scanning**: VirusTotal hash checking
- **Domain Trust**: Whitelist management
- **Real-time Updates**: Live threat intelligence

### ML Features:
- **Sender Patterns**: 11 structural and behavioral features
- **Text Analysis**: TF-IDF vectorization (500 subject + 1000 body features)
- **URL Patterns**: Count and distribution analysis
- **Ensemble Learning**: 200 Random Forest trees

## 📈 Performance Metrics

### ML Component Performance:
- **Cross-Validation Accuracy**: 97.2% (±0.8%)
- **Precision**: 95.9% (±0.8%)
- **Recall**: 99.2% (±0.6%)
- **F1-Score**: 97.5% (±0.7%)
- **ROC AUC**: 99.9%

### Hybrid System Benefits:
- **Zero False Negatives**: Blacklist catches known threats instantly
- **Low False Positives**: ML provides sophisticated pattern analysis
- **Comprehensive Coverage**: Handles both known and unknown threats
- **Real-time Performance**: Fast blacklist checks with ML fallback

## 🧪 Testing

The system includes comprehensive testing with:
- **Known Phishing**: URLs in PhishTank database
- **Unknown Phishing**: Novel attack patterns
- **Legitimate Emails**: Real emails from major services
- **Edge Cases**: Malformed emails and unusual patterns

## 🔧 Customization

### Blacklist Configuration:
- Edit `whitelist.json` to add trusted domains
- Configure API key for VirusTotal (Phishtank API key not needed)
- Adjust detection thresholds

### ML Model Tuning:
- Modify feature extraction in `model_classes.py`
- Adjust Random Forest parameters
- Retrain with new datasets from archive.zip or any other sources

## 📝 Output Format

The hybrid system provides detailed analysis:
- **Blacklist Results**: URL/attachment scanning results
- **ML Results**: Pattern analysis with confidence scores
- **Final Verdict**: Combined decision with reasoning
- **Action Items**: Clear recommendations for handling

## 🔗 Dependencies

See `requirements.txt` for the complete list of Python packages used in this project.
