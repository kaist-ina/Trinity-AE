import argparse
import os


def format_lisp_with_rules(code_string):
    """
    한 줄로 된 LISP 스타일 코드를 특정 규칙에 따라 보기 좋게 변환합니다.
    - 'load', 'index' 구문은 한 줄로 표시합니다.
    - 'store' 구문은 이름과 첫 인자까지 한 줄로 표시합니다.
    """

    # 1. 문자열을 토큰 리스트로 변환
    tokens = code_string.replace("(", " ( ").replace(")", " ) ").split()

    # 2. 토큰 리스트를 재귀적으로 파싱하여 중첩 리스트(코드 구조)로 변환
    def parse(token_list):
        if not token_list:
            raise ValueError("Unexpected end of input")
        token = token_list.pop(0)
        if token == "(":
            expr = []
            while token_list and token_list[0] != ")":
                expr.append(parse(token_list))
            if not token_list:
                raise ValueError("Unclosed parenthesis")
            token_list.pop(0)  # ')' 제거
            return expr
        elif token == ")":
            raise ValueError("Unexpected ')'")
        else:
            return token  # 단어(atom) 반환

    parsed_code = parse(tokens)

    # 3. 파싱된 코드 구조를 규칙에 맞게 문자열로 재조립
    def format_recursive(expr, indent_level=0):
        indent_str = "  " * indent_level

        # 표현식을 한 줄 문자열로 변환하는 헬퍼 함수
        def stringify(e):
            if isinstance(e, list):
                return f"({' '.join(stringify(sub) for sub in e)})"
            else:
                return str(e)

        # 표현식이 단어(atom)인 경우
        if not isinstance(expr, list):
            return indent_str + str(expr)

        # 리스트가 비어있는 경우
        if not expr:
            return indent_str + "()"

        operator = expr[0]
        args = expr[1:]

        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 변경된 부분: 'index'를 한 줄로 표시하는 규칙 추가
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        if operator in ["load", "index"]:
            return indent_str + stringify(expr)

        # 규칙 2: 'store' 구문
        if operator == "store" and args:
            header = f"{indent_str}({operator} {stringify(args[0])}"
            remaining_args = args[1:]
            if not remaining_args:
                return header + ")"

            body = [format_recursive(arg, indent_level + 1) for arg in remaining_args]
            footer = indent_str + ")"
            return "\n".join([header] + body + [footer])

        # 기본 규칙: 다른 모든 구문
        # 인자의 원래 순서를 유지하면서 처리
        formatted_args = []
        inline_args = []

        for arg in args:
            if isinstance(arg, list):
                # 이전까지의 inline args를 header에 추가
                if inline_args:
                    formatted_args.append(("inline", inline_args))
                    inline_args = []
                # list arg는 별도로 처리
                formatted_args.append(("list", arg))
            else:
                # atom은 inline args에 추가
                inline_args.append(str(arg))

        # 마지막 inline args 처리
        if inline_args:
            formatted_args.append(("inline", inline_args))

        # header 생성
        header = indent_str + "(" + str(operator)

        # 첫 번째가 inline args인 경우 header에 포함
        if formatted_args and formatted_args[0][0] == "inline":
            header += " " + " ".join(formatted_args[0][1])
            formatted_args = formatted_args[1:]

        if not formatted_args:
            return header + ")"
        else:
            body = []
            for arg_type, arg_content in formatted_args:
                if arg_type == "inline":
                    body.append(indent_str + "  " + " ".join(arg_content))
                else:  # arg_type == 'list'
                    body.append(format_recursive(arg_content, indent_level + 1))

            footer = indent_str + ")"
            return "\n".join([header] + body + [footer])

    return format_recursive(parsed_code)


parser = argparse.ArgumentParser(
    description="Format LISP-style code with specific rules"
)
parser.add_argument("--n", type=int, default=0, help="Case number to convert")
parser.add_argument("--m", type=str, default="", help="Input model type")
parser.add_argument("--t", type=str, help="Type architecture")
args = parser.parse_args()

case_file = f"./results/{args.t}/{args.t}_{args.m}_case{args.n}.txt"
# case_file = f"./results/ffn/a.txt"

with open(case_file, "r") as f:
    input_code = f.read().strip()

formatted_output = format_lisp_with_rules(input_code)
print(formatted_output)
# with open(case_file, "w") as f:
#     f.write(formatted_output)

print(f"Formatted code: {case_file}")
