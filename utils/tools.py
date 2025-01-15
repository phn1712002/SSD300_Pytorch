import os, random, string, yaml

def generate_random_name(n=4) -> str:
    """
    Tạo một cái tên ngẫu nhiên theo định dạng {name}-{exp}-{number}
    
    :param n: Số kí tự trong các name, exp, number
    :return: String tên được tạo ngẫu nhiên
    """
    # Tạo một chuỗi ngẫu nhiên tối đa n ký tự
    name = ''.join(random.choices(string.ascii_letters, k=n)).lower()
    # Tạo một chuỗi ngẫu nhiên tối đa n ký tự
    exp = ''.join(random.choices(string.ascii_letters, k=n)).lower()
    # Tạo một số ngẫu nhiên tối đa n chữ số
    number = ''.join(random.choices(string.digits, k=n))
    # Ghép thành tên theo định dạng
    random_name = f"{name}-{exp}-{number}"
    print(f"Tên được tạo {random_name}")
    return random_name

# Hàm đọc file YAML và trả về dữ liệu dạng dictionary
def load_yaml_to_dict(yaml_file_path) -> dict:
    """
    Đọc file YAML và trả về dữ liệu dạng dictionary.
    
    :param yaml_file_path: Đường dẫn file YAML.
    :return: Dictionary chứa các dữ liệu từ file YAML.
    """
    with open(yaml_file_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
    print(f"Dữ liệu đã được load từ {yaml_file_path}")
    return data

def copy_dict_excluding_keys(original_dict, keys_to_exclude):
    """
    Tạo bản sao của dictionary nhưng bỏ qua một số cặp key-value.

    :param original_dict: Dictionary gốc.
    :param keys_to_exclude: Danh sách các key cần loại bỏ.
    :return: Dictionary mới không chứa các key cần loại bỏ.
    """
    # Sử dụng dictionary comprehension để loại bỏ các key cần loại trừ
    new_dict = {key: value for key, value in original_dict.items() if key not in keys_to_exclude}
    return new_dict