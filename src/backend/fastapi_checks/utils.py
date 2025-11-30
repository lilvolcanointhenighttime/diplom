import re

from src.backend.fastapi_checks.config.log import logger
from src.backend.fastapi_checks.queries import async_query_get
from src.ml.model_provider import get_model


def format_data_for_ml_model_prediction(data: dict):
    """Преобразует данные из ФНС в формат для ML модели"""
    formatted_data = []
    
    for inn, fns_data in data.items():
        try:
            items = fns_data.get('items', [])
            if not items:
                continue
                
            first_item = items[0]
            entity = first_item.get('ЮЛ') or first_item.get('ИП')
            
            if entity and 'Позитив' in entity and 'Негатив' in entity:
                formatted_data.append({
                    'inn': inn,
                    'Позитив': entity['Позитив'],
                    'Негатив': entity['Негатив']
                })
                
        except Exception as e:
            print(f"Error processing {inn}: {e}")
            continue
    
    return formatted_data  # Возвращаем список словарей, а не словарь в списке


def model_predict_risk(data: dict):
    """Получение предсказания от ML модели"""
    try:
        # # Преобразуем данные для ML модели
        # data = format_data_for_ml_model_prediction(fns_data)
        
        if not data:
            return {"error": "No valid data to process"}
        
        # Получаем модель
        ML_model = get_model('src/ml/high_accuracy_supplier_risk_model.pkl')
        if not ML_model:
            return {"error": "ML model not available"}
        
        # Обрабатываем вложенные структуры перед предсказанием
        # processed_data = []
        # for item in data:
        #     processed_item = {
        #         "Позитив": process_nested_dict(item["Позитив"]),
        #         "Негатив": process_nested_dict(item["Негатив"])
        #     }
        #     processed_data.append(processed_item)

        result = []
        # print(data.items())
        for inn, rep_data in data.items():
            ml_prediction = {}
            ml_prediction["inn"] = inn  
            ml_prediction["data"] = ML_model.predict_risk(rep_data)
            result.append(ml_prediction)
        return result
        
    except Exception as e:
        logger.error(f"Error in ML prediction: {e}")
        return {"error": str(e)}


def process_nested_dict(data):
    """Обрабатывает вложенные словари, преобразуя их в плоскую структуру"""
    result = {}
    for value, value in data.items():
        if isinstance(value, dict):
            for sub_value, sub_value in value.items():
                # Пропускаем сложные вложенные структуры
                if not isinstance(sub_value, (dict, list)):
                    result[f"{value}_{sub_value}"] = sub_value
        elif not isinstance(value, (dict, list)):
            # Сохраняем только простые значения
            result[value] = value
    return result


async def reputation_check(entyties: list[dict], REPUTATION_API_KEY: str):
    logger.info(f"Getting information from reputation")
    check_results: dict = {}
    try:
        for entity in entyties:
            inn = entity["inn"]
            kpp = entity["kpp"]
            
            entity_data = await reputation_entity_id(inn, kpp, REPUTATION_API_KEY)
            if not entity_data["Items"]:
                check_results[inn] = None
                continue

            entity_id = entity_data["Items"][0]["Id"]
            entity_type = entity_data["Items"][0]["Type"]

            data = await async_query_get(
                f"https://api.reputation.ru/api/v3/express-check/report",
                headers={
                    "Authorization":REPUTATION_API_KEY,
                    "Content-Type":"application/json",
                },
                params={
                    "entityId": entity_id,
                    "entityType": entity_type,
                    "version": "Short",
                    "documentType": "Json"
                }
            )
            
            check_results[inn] = data

    except Exception as e:
        logger.error(f"Error while checking INNs: {e}")
        raise e
    
    return check_results

async def reputation_entity_id(inn: str, kpp: str, REPUTATION_API_KEY: str):
    data = await async_query_get(
        url="https://api.reputation.ru/api/v1/Entities/id",
        headers={
            "Authorization": REPUTATION_API_KEY,
            "Content-Type": "application/json"
        },
        params={
            "inn": inn,
            # "kpp": kpp
        }
    )
    return data

import re

def extract_features_from_rep_data(rep_data: dict):
    """Извлекает признаки из списка факторов компании"""
    features = {}

    # Определяем признаки по умолчанию
    default_features = {
        'has_losses': 0, 'zero_revenue': 0, 'no_reporting': 0, 'has_profit': 0,
        'company_age': 0.0,  # Теперь это float
        'recent_activity': 0, 'tax_debt': 0, 'mass_address': 0,
        'unreliable_address': 0, 'mass_manager': 0, 'mass_shareholder': 0,
        'has_bailiff_cases': 0, 'bankruptcy_case': 0, 'arbitration_cases': 0,
        'government_owned': 0, 'has_bank_license': 0, 'has_insurance_license': 0
    }

    for inn, data in rep_data.items():
        # Инициализируем пустой словарь для каждой компании
        features[inn] = {}
        
        for i, factor in enumerate(data["Factors"]):
            factor_type = factor['Type']
            score = factor['Score']
            description = factor.get('Description', '')
            name = factor.get('Name', '')
            print(description)
            
            # === ФИНАНСОВЫЕ ПОКАЗАТЕЛИ ===
            if factor_type == 'Finance':
                if 'убытки' in str(description):
                    features[inn]['has_losses'] = 1
                elif 'выручка равна нулю' in str(description):
                    features[inn]['zero_revenue'] = 1
                elif 'отчетность отсутствует' in str(description):
                    features[inn]['no_reporting'] = 1
                    
            elif factor_type == 'FinanceProfit':
                features[inn]['has_profit'] = 1
                
            # === ВОЗРАСТ КОМПАНИИ ===
            elif factor_type == 'Lifetime':
                # Парсим "Организация существует 9 лет и 2 месяца."
                age_match = re.search(r'существует (\d+) лет(?: и (\d+) месяцев)?', name)
                if age_match:
                    years = int(age_match.group(1))
                    months = int(age_match.group(2)) if age_match.group(2) else 0
                    features[inn]['company_age'] = years + months/12
            
            # === АКТИВНОСТЬ ===
            elif factor_type == 'Activity':
                features[inn]['recent_activity'] = 1
                
            # === НАЛОГОВАЯ ЗАДОЛЖЕННОСТЬ ===
            elif factor_type == 'Debtor':
                features[inn]['tax_debt'] = 0 if score == 'Positive' else 1
                
            # === МАССОВЫЙ АДРЕС ===
            elif factor_type == 'MassAddress':
                features[inn]['mass_address'] = 0 if score == 'Positive' else 1
                
            # === НЕДОСТОВЕРНЫЙ АДРЕС ===
            elif factor_type == 'AddressUnreliability':
                features[inn]['unreliable_address'] = 1 if score == 'Attention' else 0
                
            # === МАССОВЫЙ РУКОВОДИТЕЛЬ ===
            elif factor_type == 'MassManager':
                features[inn]['mass_manager'] = 1 if score == 'Attention' else 0
                
            # === ИСПОЛНИТЕЛЬНЫЕ ПРОИЗВОДСТВА ===
            elif factor_type == 'BailiffCurrent':
                features[inn]['has_bailiff_cases'] = 1 if score == 'Attention' else 0
                
            # === ДЕЛО О БАНКРОТСТВЕ ===
            elif factor_type == 'Bankruptcy':
                features[inn]['bankruptcy_case'] = 1 if score == 'Attention' else 0
                
            # === АРБИТРАЖНЫЕ ДЕЛА ===
            elif factor_type == 'ArbitrationSumPlaintiff':
                features[inn]['arbitration_cases'] = 1
                
            elif factor_type == 'ArbitrationSumDefendant':
                features[inn]['arbitration_cases'] = 1
                
            # === СПЕЦИАЛЬНЫЕ СТАТУСЫ ===
            elif factor_type == 'GovernmentOwned':
                features[inn]['government_owned'] = 1
            elif factor_type == 'BankLicense':
                features[inn]['has_bank_license'] = 1
            elif factor_type == 'InsuranceLicense':
                features[inn]['has_insurance_license'] = 1
        
        # ЗАПОЛНЯЕМ ОТСУТСТВУЮЩИЕ ПРИЗНАКИ ЗНАЧЕНИЯМИ ПО УМОЛЧАНИЮ
        for key in default_features:
            if key not in features[inn]:
                features[inn][key] = default_features[key]
    
    return features