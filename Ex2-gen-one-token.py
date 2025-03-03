import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

def show_tokenization(inputs, tokenizer):
    """토큰화된 결과를 데이터프레임 형태로 보여주는 함수
    Args:
        inputs: 토큰화된 입력값
        tokenizer: 사용중인 토크나이저
    Returns:
        DataFrame: 토큰 ID와 해당하는 실제 토큰을 보여주는 데이터프레임
    """
    return pd.DataFrame(
        [(id, tokenizer.decode(id)) for id in inputs["input_ids"][0]],
        columns=["id", "token"],
    )

def show_next_token_choices(probabilities, tokenizer, top_n=5):
    """다음 토큰의 확률 분포를 보여주는 함수
    Args:
        probabilities: 각 토큰별 확률값
        tokenizer: 사용중인 토크나이저
        top_n: 상위 몇 개의 확률을 보여줄지 지정
    Returns:
        DataFrame: 상위 n개의 토큰과 그 확률을 보여주는 데이터프레임
    """
    return pd.DataFrame(
        [
            (id, tokenizer.decode(id), p.item())
            for id, p in enumerate(probabilities)
            if p.item()
        ],
        columns=["id", "token", "p"],
    ).sort_values("p", ascending=False)[:top_n]

def main():
    # GPT-2 모델과 토크나이저 로드
    print("모델과 토크나이저를 로딩중...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # 초기 텍스트 설정
    text = "Udacity is the best place to learn about generative"
    print("\n초기 텍스트:", text)

    # 텍스트 토큰화
    inputs = tokenizer(text, return_tensors="pt")
    
    # 토큰화 결과 출력
    print("\n토큰화 결과:")
    print(show_tokenization(inputs, tokenizer))

    # 다음 토큰 예측을 위한 확률 계산
    print("\n다음 토큰 예측 중...")
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
        probabilities = torch.nn.functional.softmax(logits[0], dim=-1)

    # 상위 5개 토큰 선택지 출력
    print("\n다음 토큰 상위 5개 선택지:")
    print(show_next_token_choices(probabilities, tokenizer))

    # 가장 확률이 높은 토큰 선택 및 텍스트에 추가
    next_token_id = torch.argmax(probabilities).item()
    print(f"\n선택된 다음 토큰 ID: {next_token_id}")
    print(f"선택된 다음 토큰: {tokenizer.decode(next_token_id)}")
    
    # 전체 문장 생성 예시
    print("\n전체 문장 생성 예시:")
    initial_text = "Once upon a time, generative models"
    inputs = tokenizer(initial_text, return_tensors="pt")
    
    # generate 메소드를 사용하여 텍스트 생성
    output = model.generate(
        **inputs, 
        max_length=100, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 생성된 텍스트 출력
    generated_text = tokenizer.decode(output[0])
    print("\n생성된 텍스트:")
    print(generated_text)

if __name__ == "__main__":
    main()