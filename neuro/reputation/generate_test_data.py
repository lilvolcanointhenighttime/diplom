import random
import json
from typing import List, Dict, Tuple

def generate_test_data(num_samples: int, random_seed: int = 42, save_path: str = None) -> List[Dict]:
    """
    Генерирует тестовые данные для обучения модели с классификацией рисков
    
    Args:
        num_samples (int): количество тестовых записей
        random_seed (int): seed для воспроизводимости результатов
        save_path (str, optional): путь для сохранения в JSON файл
    
    Returns:
        List[Dict]: список словарей с тестовыми данными и классификацией рисков
    """
    random.seed(random_seed)
    
    test_data = []
    
    # Базовые значения по умолчанию
    default_features = {
        'has_losses': 0, 'zero_revenue': 0, 'no_reporting': 0, 'has_profit': 0,
        'company_age': 0.0, 'recent_activity': 0, 'tax_debt': 0, 'mass_address': 0,
        'unreliable_address': 0, 'mass_manager': 0, 'mass_shareholder': 0,
        'has_bailiff_cases': 0, 'bankruptcy_case': 0, 'arbitration_cases': 0,
        'government_owned': 0, 'has_bank_license': 0, 'has_insurance_license': 0
    }
    
    for i in range(num_samples):
        # Создаем копию значений по умолчанию
        features = default_features.copy()
        
        # Генерируем реалистичные значения признаков
        features['company_age'] = round(random.uniform(0.5, 30.0), 1)  # от 0.5 до 30 лет
        
        # Вероятности для бинарных признаков
        features['recent_activity'] = random.choices([0, 1], weights=[0.2, 0.8])[0]
        features['has_losses'] = random.choices([0, 1], weights=[0.7, 0.3])[0]
        features['tax_debt'] = random.choices([0, 1], weights=[0.8, 0.2])[0]
        features['mass_address'] = random.choices([0, 1], weights=[0.9, 0.1])[0]
        features['unreliable_address'] = random.choices([0, 1], weights=[0.85, 0.15])[0]
        features['has_bailiff_cases'] = random.choices([0, 1], weights=[0.75, 0.25])[0]
        features['bankruptcy_case'] = random.choices([0, 1], weights=[0.95, 0.05])[0]
        features['zero_revenue'] = random.choices([0, 1], weights=[0.9, 0.1])[0]
        features['has_profit'] = random.choices([0, 1], weights=[0.4, 0.6])[0]
        features['mass_manager'] = random.choices([0, 1], weights=[0.85, 0.15])[0]
        features['mass_shareholder'] = random.choices([0, 1], weights=[0.9, 0.1])[0]
        features['arbitration_cases'] = random.choices([0, 1], weights=[0.7, 0.3])[0]
        features['government_owned'] = random.choices([0, 1], weights=[0.95, 0.05])[0]
        features['has_bank_license'] = random.choices([0, 1], weights=[0.98, 0.02])[0]
        features['has_insurance_license'] = random.choices([0, 1], weights=[0.98, 0.02])[0]
        
        # Добавляем логические зависимости
        if features['has_losses'] == 1:
            features['has_profit'] = 0
        if features['zero_revenue'] == 1:
            features['has_profit'] = 0
            features['has_losses'] = 0  # нет выручки - нет и убытков/прибыли
        
        # КЛАССИФИКАЦИЯ РИСКА
        risk_level, risk_score = classify_risk(features)
        features['risk_level'] = risk_level
        features['risk_score'] = risk_score
        
        test_data.append(features)
    
    # Сохраняем в файл если указан путь
    if save_path:
        save_test_data_to_json(test_data, save_path)
    
    return test_data

def classify_risk(features: Dict) -> Tuple[str, int]:
    """
    Классифицирует уровень риска на основе признаков компании
    
    Args:
        features (Dict): словарь с признаками компании
    
    Returns:
        Tuple[str, int]: (уровень_риска, балл_риска)
    """
    risk_score = 0
    
    # Веса для различных факторов риска
    weights = {
        # Критические факторы (высокий вес)
        'bankruptcy_case': 10,
        'tax_debt': 8,
        'has_bailiff_cases': 7,
        'unreliable_address': 9,
        
        # Серьезные факторы (средний вес)
        'has_losses': 5,
        'zero_revenue': 6,
        'no_reporting': 6,
        'mass_address': 4,
        'mass_manager': 4,
        'mass_shareholder': 4,
        
        # Умеренные факторы (низкий вес)
        'arbitration_cases': 3,
        
        # Позитивные факторы (уменьшают риск)
        'has_profit': -4,
        'recent_activity': -2,
        'company_age': -0.1,  # за каждый год возраста
        'government_owned': -3,
        'has_bank_license': -5,
        'has_insurance_license': -5
    }
    
    # Вычисляем общий балл риска
    for feature, weight in weights.items():
        if feature == 'company_age':
            # Для возраста используем специальную логику
            risk_score += features[feature] * weight
        else:
            risk_score += features[feature] * weight
    
    # Определяем уровень риска
    if risk_score <= 5:
        risk_level = "МИНИМАЛЬНЫЙ РИСК"
    elif risk_score <= 15:
        risk_level = "УМЕРЕННЫЙ РИСК"
    else:
        risk_level = "КРИТИЧЕСКИЙ РИСК"
    
    return risk_level, round(risk_score, 1)

def save_test_data_to_json(test_data: List[Dict], file_path: str):
    """
    Сохраняет тестовые данные в JSON файл
    
    Args:
        test_data (List[Dict]): список с тестовыми данными
        file_path (str): путь к файлу для сохранения
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        print(f"Данные успешно сохранены в файл: {file_path}")
        print(f"Всего записей: {len(test_data)}")
        
        # Статистика по рискам
        risk_counts = {}
        for item in test_data:
            risk_level = item['risk_level']
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        print("Распределение рисков:")
        for risk, count in risk_counts.items():
            print(f"  {risk}: {count} записей")
            
    except Exception as e:
        print(f"Ошибка при сохранении файла: {e}")

# Упрощенная версия с классификацией
def generate_simple_test_data(num_samples: int, save_path: str = None) -> List[Dict]:
    """
    Упрощенная версия для генерации тестовых данных с классификацией рисков
    """
    test_data = []
    
    # Предопределенные шаблоны для разных типов компаний
    templates = [
        # Стабильная компания (МИНИМАЛЬНЫЙ РИСК)
        {
            'has_losses': 0, 'zero_revenue': 0, 'no_reporting': 0, 'has_profit': 1,
            'company_age': 15.0, 'recent_activity': 1, 'tax_debt': 0, 'mass_address': 0,
            'unreliable_address': 0, 'mass_manager': 0, 'mass_shareholder': 0,
            'has_bailiff_cases': 0, 'bankruptcy_case': 0, 'arbitration_cases': 0,
            'government_owned': 0, 'has_bank_license': 0, 'has_insurance_license': 0
        },
        # Компания с проблемами (КРИТИЧЕСКИЙ РИСК)
        {
            'has_losses': 1, 'zero_revenue': 0, 'no_reporting': 0, 'has_profit': 0,
            'company_age': 3.5, 'recent_activity': 1, 'tax_debt': 1, 'mass_address': 1,
            'unreliable_address': 1, 'mass_manager': 0, 'mass_shareholder': 0,
            'has_bailiff_cases': 1, 'bankruptcy_case': 0, 'arbitration_cases': 1,
            'government_owned': 0, 'has_bank_license': 0, 'has_insurance_license': 0
        },
        # Новая компания (УМЕРЕННЫЙ РИСК)
        {
            'has_losses': 0, 'zero_revenue': 1, 'no_reporting': 0, 'has_profit': 0,
            'company_age': 0.8, 'recent_activity': 1, 'tax_debt': 0, 'mass_address': 0,
            'unreliable_address': 0, 'mass_manager': 0, 'mass_shareholder': 0,
            'has_bailiff_cases': 0, 'bankruptcy_case': 0, 'arbitration_cases': 0,
            'government_owned': 0, 'has_bank_license': 0, 'has_insurance_license': 0
        },
        # Компания с лицензией (МИНИМАЛЬНЫЙ РИСК)
        {
            'has_losses': 0, 'zero_revenue': 0, 'no_reporting': 0, 'has_profit': 1,
            'company_age': 20.0, 'recent_activity': 1, 'tax_debt': 0, 'mass_address': 0,
            'unreliable_address': 0, 'mass_manager': 0, 'mass_shareholder': 0,
            'has_bailiff_cases': 0, 'bankruptcy_case': 0, 'arbitration_cases': 0,
            'government_owned': 1, 'has_bank_license': 1, 'has_insurance_license': 0
        }
    ]
    
    for i in range(num_samples):
        template = random.choice(templates)
        
        # Создаем копию шаблона с небольшими вариациями
        features = template.copy()
        features['company_age'] = round(features['company_age'] + random.uniform(-2, 2), 1)
        
        # Классификация риска
        risk_level, risk_score = classify_risk(features)
        features['risk_level'] = risk_level
        features['risk_score'] = risk_score
        
        test_data.append(features)
    
    # Сохраняем в файл если указан путь
    if save_path:
        save_test_data_to_json(test_data, save_path)
    
    return test_data

test_data = generate_test_data(1500, save_path='test_data_with_risks.json')