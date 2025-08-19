"""
Link documents: [https://docs.python.org/3/library/argparse.html]

Ví dụ chi tiết về cách sử dụng argparse trong Python.
argparse là một module tiêu chuẩn giúp phân tích các Command-line arguments.

Nó sẽ giúp bạn: 
    - Định nghĩa các tham số (argument) mà script của bạn nhận.
    - Tự động parse (phân tích) tham số đó từ sys.argv.
    - Tự động tạo phần help (-h/--help) cho người dùng.
    - Xác thực kiểu dữ liệu, giá trị mặc định, bắt buộc hay tùy chọn.

DETAIL

1. Lấy hướng dẫn sử dụng: python code_explanation/argparse_detail.py --help

2. add_argument() method:
    - name or flags: "foo", "-f", "--foo". positional/optional arguments
    - action: Xác định hành động khi gặp đối số.
    - nargs: Số lượng giá trị đi kèm option hoặc positional.
        Số nguyên n: chính xác n giá trị.
        "?": 0 hoặc 1 giá trị.
        "*": 0 hoặc nhiều giá trị.
        "+": 1 hoặc nhiều giá trị.
        argparse.REMAINDER: lấy toàn bộ phần còn lại của dòng lệnh.
    - default
    - type: ép kiểu
    - choices: Danh sách hoặc tuple các giá trị hợp lệ. Nếu nhập sai → báo lỗi và hiển thị các giá trị hợp lệ.
    - help: mô tả ngắn/súc tích

3. How to run?
    python code_explanation/argparse_detail.py [positional_args...] [options...]
        
        - positional_args: đối số vị trí, bắt buộc theo thứ tự (trừ khi dùng nargs='?' hoặc *).
        - options: tùy chọn, bắt đầu bằng - hoặc --, thứ tự linh hoạt, có thể không truyền (trừ khi required=True).
        - Ví dụ: python code_explanation/argparse_detail.py 1 2 --operation add
"""


import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Ví dụ về argparse: tính toán tổng hoặc hiệu của hai số."
    )

    # Thêm các arguments vào parser
    # --a và --b là hai số nguyên bắt buộc
    parser.add_argument("-a", "--a", type=int,
                        required=True, help="Số nguyên thứ nhất")
    parser.add_argument("-b", "--b", type=int,
                        required=True, help="Số nguyên thứ hai")

    # --operation là một lựa chọn, mặc định là 'add'
    parser.add_argument("-o", "--operation", choices=["add", "sub"], default="add",
                        help="Phép toán thực hiện: 'add' (cộng) hoặc 'sub' (trừ). Mặc định là 'add'.")

    # Phân tích các đối số dòng lệnh
    args = parser.parse_args()

    # In ra các giá trị đã nhận được
    print(f"Giá trị a: {args.a}")
    print(f"Giá trị b: {args.b}")
    print(f"Phép toán: {args.operation}")

    # Thực hiện phép toán dựa trên lựa chọn
    if args.operation == "add":
        result = args.a + args.b
        print(f"Kết quả cộng: {result}")
    else:
        result = args.a - args.b
        print(f"Kết quả trừ: {result}")


if __name__ == "__main__":
    main()
