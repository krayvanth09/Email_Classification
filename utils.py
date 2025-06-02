"""
Utility functions and helpers
"""

import logging
import os
import sys
from typing import Dict, Any, List
import json

def setup_logging(level: str = "INFO"):
    """
    Setup logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def validate_api_response(response_data: Dict[str, Any]) -> bool:
    """
    Validate API response format
    
    Args:
        response_data: Response dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        'input_email_body',
        'list_of_masked_entities',
        'masked_email',
        'category_of_the_email'
    ]
    
    # Check all required fields are present
    for field in required_fields:
        if field not in response_data:
            return False
    
    # Check list_of_masked_entities format
    entities = response_data.get('list_of_masked_entities', [])
    if not isinstance(entities, list):
        return False
    
    for entity in entities:
        if not isinstance(entity, dict):
            return False
        
        required_entity_fields = ['position', 'classification', 'entity']
        for field in required_entity_fields:
            if field not in entity:
                return False
        
        # Check position format [start, end]
        position = entity.get('position')
        if not isinstance(position, list) or len(position) != 2:
            return False
        
        if not all(isinstance(x, int) for x in position):
            return False
    
    return True

def create_test_email_samples() -> List[Dict[str, str]]:
    """
    Create test email samples for validation
    
    Returns:
        List of test email dictionaries
    """
    return [
        {
            'email': 'Subject: System Crash\n\nDear Support, Our system crashed unexpectedly. My name is John Doe and you can reach me at john.doe@company.com or call +1-555-123-4567.',
            'expected_category': 'Incident'
        },
        {
            'email': 'Subject: Information Request\n\nHello, I would like to request information about your services. My name is Jane Smith, email: jane.smith@email.com.',
            'expected_category': 'Request'
        },
        {
            'email': 'Subject: Account Update\n\nI need to change my account settings. Please help me update my profile. Contact: alice@domain.com.',
            'expected_category': 'Change'
        },
        {
            'email': 'Subject: Ongoing Issue\n\nWe have been experiencing persistent problems with data sync. This issue keeps recurring.',
            'expected_category': 'Problem'
        }
    ]

def mask_sensitive_info_in_logs(text: str) -> str:
    """
    Mask sensitive information in log messages
    
    Args:
        text: Text that might contain sensitive information
        
    Returns:
        Text with sensitive information masked
    """
    import re
    
    # Mask email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Mask phone numbers
    text = re.sub(r'[\+]?[\d\-\(\)\s]{10,}', '[PHONE]', text)
    
    # Mask credit card numbers
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)
    
    return text

def check_model_file_exists(model_path: str) -> bool:
    """
    Check if model file exists and is readable
    
    Args:
        model_path: Path to model file
        
    Returns:
        True if file exists and is readable
    """
    try:
        return os.path.exists(model_path) and os.access(model_path, os.R_OK)
    except Exception:
        return False

def get_environment_info() -> Dict[str, Any]:
    """
    Get environment information for debugging
    
    Returns:
        Dictionary with environment details
    """
    import platform
    import sys
    
    return {
        'python_version': sys.version,
        'platform': platform.platform(),
        'working_directory': os.getcwd(),
        'environment_variables': {
            key: value for key, value in os.environ.items() 
            if key.startswith(('API_', 'MODEL_', 'DATA_'))
        }
    }

def format_entity_for_response(entity: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format entity for API response
    
    Args:
        entity: Raw entity dictionary
        
    Returns:
        Formatted entity for response
    """
    return {
        'position': [entity['start'], entity['end']],
        'classification': entity['label'],
        'entity': entity['text']
    }

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
    
    # Return default config
    return {
        'model_path': 'email_classifier_model.pkl',
        'log_level': 'INFO',
        'api_port': 8000,
        'max_email_length': 10000
    }
