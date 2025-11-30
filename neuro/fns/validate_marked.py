import json

def quick_simplify():
    """
    Быстрое упрощение датасета без дополнительных проверок
    """
    input_file = "quick_marked_checks.json"
    output_file = "simplified_dataset.json"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    simplified = []
    
    for item in data:
        if item.get('status') == 'success':
            for company in item.get('data', {}).get('items', []):
                if 'ЮЛ' in company:
                    new_record = {
                        'Позитив': company['ЮЛ']['Позитив'],
                        'Негатив': company['ЮЛ']['Негатив']
                    }
                    
                    # Добавляем рекомендации если есть
                    if 'рекомендация' in company:
                        new_record['рекомендация'] = company['рекомендация']
                    
                    simplified.append(new_record)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simplified, f, ensure_ascii=False, indent=2)
    
    print(f"Создан упрощенный файл: {output_file}")
    print(f"Записей: {len(simplified)}")

def frequency_analysis():
    """
    Анализ с подсчетом частоты встречаемости
    """
    input_file = "simplified_dataset.json"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    from collections import Counter
    
    rec_counter = Counter()
    
    for item in data:
        if 'рекомендация' in item:
            rec_counter[item['рекомендация']] += 1
    
    print("\nЧАСТОТА РЕКОМЕНДАЦИЙ:")
    for rec, count in rec_counter.most_common():
        print(f"{rec}: {count} раз")


# Для быстрого запуска
if __name__ == "__main__":
    quick_simplify()
    frequency_analysis()