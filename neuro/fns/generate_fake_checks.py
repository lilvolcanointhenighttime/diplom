import json
import random

def generate_inn_10():
    """Генерация случайного 12-значного ИНН"""
    inn = ''.join([str(random.randint(0, 9)) for _ in range(10)])
    return inn

def generate_fns_check_results():
    """Генерирует 100 проверок ФНС в заданном формате"""
    
    # Список реальных ИНН российских компаний
    inns = [
        "7736050003", "7707033437", "7744001497", "7702070139", "7830002293",
        "7710030411", "7708514824", "7718249396", "1660048363", "2315117051",
        "2721135590", "7702204400", "3664069397", "1653003353", "6312101021",
        "7726575524", "2310021996", "7706092528", "5024054155", "7709285447",
        "7710498023", "7730170714", "7705182000", "7708707667", "7712040123",
        "7736200703", "7709929321", "7708230001", "7713056770", "7731405640",
        "7706295654", "7715984788", "7713076301", "7709444666", "7710021994",
        "7730023514", "7705031674", "7714083247", "7707058517", "7712458492",
        "7701904633", "7710567910", "7702202400", "7710026577", "7709876543",
        "7734232345", "7705123456", "7715987654", "7709123456", "7712345678",
        "7723456789", "7734567890", "7745678901", "7756789012", "7767890123",
        "7778901234", "7789012345", "7790123456", "7801234567", "7812345678",
        "7823456789", "7834567890", "7845678901", "7856789012", "7867890123",
        "7878901234", "7889012345", "7890123456", "7901234567", "7912345678",
        "7923456789", "7934567890", "7945678901", "7956789012", "7967890123",
        "7978901234", "7989012345", "7990123456", "8001234567", "8012345678",
        "8023456789", "8034567890", "8045678901", "8056789012", "8067890123",
        "8078901234", "8089012345", "8090123456", "8101234567", "8112345678",
        "8123456789", "8134567890", "8145678901", "8156789012", "8167890123"
    ]
    
    results = []
    
    # Позитивные факторы
    positive_factors = [
        {"Лицензии": "Есть", "Филиалы": "Есть", "КапБолее50тыс": "Да"},
        {"Лицензии": "Есть", "Филиалы": "Нет", "КапБолее50тыс": "Да"},
        {"Лицензии": "Нет", "Филиалы": "Есть", "КапБолее50тыс": "Да"},
        {"Лицензии": "Есть", "Филиалы": "Есть", "КапБолее50тыс": "Нет"},
        {"Лицензии": "Нет", "Филиалы": "Нет", "КапБолее50тыс": "Да"}
    ]
    
    positive_texts = [
        "Есть лицензии ({licenses} шт.); Есть филиалы ({branches} шт.); Уставный капитал {capital} тыс. руб.",
        "Лицензии ({licenses} шт.) имеются; Филиальная сеть ({branches} шт.); Капитализация {capital} тыс. руб.",
        "Наличие {licenses} лицензий; {branches} филиалов; Размер уставного капитала {capital} тыс. руб."
    ]
    
    # Негативные факторы
    negative_factors = [
        {"РеестрМассАдрес": "Нет", "МассАдрес": "Нет"},
        {"РеестрМассАдрес": "Да ({count1} организаций)", "МассАдрес": "Да ({count2} юрлиц)"},
        {"НалоговыеНарушения": "Есть", "СудебныеДела": "Нет"},
        {"СудебныеДела": "Есть ({count} дел)", "Банкротство": "Нет"},
        {"РеестрМассАдрес": "Нет", "НалоговыеНарушения": "Есть ({count} нарушений)"},
        {"МассАдрес": "Да ({count} юрлиц)", "СудебныеДела": "Есть"},
        {"Банкротство": "Признаки", "РеестрМассАдрес": "Да"}
    ]
    
    negative_texts = [
        "В реестре массовых адресов ({count1} юрлиц, в БД найдено - {count2} юрлиц)",
        "Налоговые нарушения ({count} случаев)",
        "Судебные дела в производстве ({count} шт.)",
        "Признаки банкротства; Наличие массового адреса",
        "Нарушений не выявлено",
        "Массовый адрес ({count} организаций)",
        "Налоговые и судебные риски присутствуют"
    ]
    inn_list = [generate_inn_10() for _ in range(300)]

    for inn in enumerate(inn_list):
        # Выбираем случайные факторы
        positive = random.choice(positive_factors).copy()
        negative = random.choice(negative_factors).copy()
        
        # Генерируем текстовые описания
        licenses_count = random.randint(1, 25) if positive.get("Лицензии") == "Есть" else 0
        branches_count = random.randint(1, 150) if positive.get("Филиалы") == "Есть" else 0
        capital = random.randint(50000, 500000000)
        
        positive_text = random.choice(positive_texts).format(
            licenses=licenses_count,
            branches=branches_count,
            capital=capital
        )
        
        # Для негативных текстов
        count1 = random.randint(1, 20)
        count2 = random.randint(1, 25)
        count = random.randint(1, 10)
        
        negative_text = random.choice(negative_texts).format(
            count1=count1,
            count2=count2,
            count=count
        )
        
        # Заменяем плейсхолдеры в негативных факторах
        for key in negative:
            if "{count1}" in str(negative[key]):
                negative[key] = negative[key].format(count1=count1)
            elif "{count2}" in str(negative[key]):
                negative[key] = negative[key].format(count2=count2)
            elif "{count}" in str(negative[key]):
                negative[key] = negative[key].format(count=count)
        
        # Создаем запись
        result = {
            "inn": inn,
            "status": "success",
            "data": {
                "items": [
                    {
                        "ЮЛ": {
                            "ОГРН": f"1{random.randint(0, 9)}{random.randint(10, 99)}{random.randint(1000000, 9999999)}",
                            "ИНН": inn,
                            "Позитив": {
                                **positive,
                                "Текст": positive_text
                            },
                            "Негатив": {
                                **negative,
                                "Текст": negative_text
                            }
                        }
                    }
                ]
            }
        }
        
        results.append(result)
    
    return results

# Генерируем и сохраняем данные
results = generate_fns_check_results()

# Сохраняем в файл
with open('fake_checks_100.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Сгенерировано {len(results)} проверок и сохранено в файл 'fns_checks_100.json'")

# Выводим статистику
positive_stats = {
    "Лицензии": 0,
    "Филиалы": 0,
    "КапБолее50тыс": 0
}

negative_stats = {
    "РеестрМассАдрес": 0,
    "МассАдрес": 0,
    "НалоговыеНарушения": 0,
    "СудебныеДела": 0,
    "Банкротство": 0
}

for result in results:
    positive = result["data"]["items"][0]["ЮЛ"]["Позитив"]
    negative = result["data"]["items"][0]["ЮЛ"]["Негатив"]
    
    for key in positive_stats:
        if key in positive and positive[key] in ["Есть", "Да"]:
            positive_stats[key] += 1
    
    for key in negative_stats:
        if key in negative and "Да" in str(negative[key]) or "Есть" in str(negative[key]):
            negative_stats[key] += 1

print("\nСтатистика позитивных факторов:")
for factor, count in positive_stats.items():
    print(f"  {factor}: {count}")

print("\nСтатистика негативных факторов:")
for factor, count in negative_stats.items():
    print(f"  {factor}: {count}")