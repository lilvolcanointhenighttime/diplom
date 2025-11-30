import json
import re

def simple_mark_company(company_data):
    """
    Упрощенная версия анализа компании
    """
    p = company_data['Позитив']
    n = company_data['Негатив']
    
    # Позитивные факторы
    pos_count = 0
    if p['Лицензии'] == 'Есть':
        pos_count += 1
    if p['Филиалы'] == 'Есть':
        pos_count += 1
    if p['КапБолее50тыс'] == 'Да':
        pos_count += 1
    
    # Негативные факторы
    neg_count = 0
    if 'Банкротство' in n and 'Признаки' in str(n['Банкротство']):
        neg_count += 2
    if n.get('РеестрМассАдрес') == 'Да' or n.get('МассАдрес') == 'Да':
        neg_count += 2
    if n.get('НалоговыеНарушения') == 'Есть':
        neg_count += 1
    if n.get('СудебныеДела') == 'Есть':
        neg_count += 1
    
    # Определение категории
    if neg_count >= 3:
        return "Критический риск", "Множество серьезных негативных факторов"
    elif neg_count >= 2:
        return "Повышенный риск", "Значительные негативные факторы"
    elif pos_count >= 2 and neg_count == 0:
        return "Высокая надежность", "Сильные позитивные показатели при отсутствии негативных"
    else:
        return "Условная надежность", "Смешанные показатели надежности"

def quick_mark_dataset():
    """
    Быстрая разметка датасета
    """
    with open('fake_checks_300.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        if item['status'] == 'success':
            for company_item in item['data']['items']:
                if 'ЮЛ' in company_item:
                    rec, just = simple_mark_company(company_item['ЮЛ'])
                    company_item['рекомендация'] = rec
                    company_item['обоснование'] = just
    
    with open('quick_marked_checks.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("Быстрая разметка завершена!")

# Для быстрого запуска используйте:
quick_mark_dataset()