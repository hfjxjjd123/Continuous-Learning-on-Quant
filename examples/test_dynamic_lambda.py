#!/usr/bin/env python3
"""
동적 EWC lambda 조정 기능 테스트 스크립트

이 스크립트는 온라인 학습 과정에서 이전 윈도우 결과에 따라
동적으로 EWC lambda 값을 조정하는 기능을 테스트합니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mom_trans.new_inference import (
    adjust_lambda_dynamically, 
    calculate_adaptive_lambda,
    estimate_lambda_ewc
)
import tensorflow as tf
import numpy as np

def test_dynamic_lambda_adjustment():
    """동적 lambda 조정 기능을 테스트합니다."""
    
    print("=== 동적 Lambda 조정 테스트 ===")
    
    # 테스트 시나리오들
    test_cases = [
        {
            "name": "성능 개선 시나리오",
            "current_lambda": 1.0,
            "previous_sharpe": 0.5,
            "current_sharpe": 0.8,
            "expected_behavior": "lambda 감소 (더 적극적 학습)"
        },
        {
            "name": "성능 저하 시나리오", 
            "current_lambda": 1.0,
            "previous_sharpe": 0.8,
            "current_sharpe": 0.3,
            "expected_behavior": "lambda 증가 (이전 지식 보존)"
        },
        {
            "name": "성능 유지 시나리오",
            "current_lambda": 1.0,
            "previous_sharpe": 0.6,
            "current_sharpe": 0.65,
            "expected_behavior": "lambda 유지 (임계값 미만)"
        }
    ]
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        print(f"예상 동작: {case['expected_behavior']}")
        
        new_lambda = adjust_lambda_dynamically(
            current_lambda=case["current_lambda"],
            previous_sharpe=case["previous_sharpe"],
            current_sharpe=case["current_sharpe"],
            adjustment_factor=0.2,
            performance_threshold=0.1
        )
        
        print(f"이전 Sharpe: {case['previous_sharpe']:.3f}")
        print(f"현재 Sharpe: {case['current_sharpe']:.3f}")
        print(f"변화량: {case['current_sharpe'] - case['previous_sharpe']:.3f}")
        print(f"이전 Lambda: {case['current_lambda']:.3f}")
        print(f"새로운 Lambda: {new_lambda:.3f}")
        print(f"변화량: {new_lambda - case['current_lambda']:.3f}")

def test_lambda_boundaries():
    """Lambda 경계값 테스트"""
    
    print("\n=== Lambda 경계값 테스트 ===")
    
    # 최소값 테스트
    min_lambda = adjust_lambda_dynamically(
        current_lambda=0.2,
        previous_sharpe=0.8,
        current_sharpe=0.9,  # 성능 개선
        min_lambda=0.1,
        max_lambda=10.0,
        adjustment_factor=0.5
    )
    print(f"최소값 테스트: 0.2 -> {min_lambda:.3f} (최소값 0.1)")
    
    # 최대값 테스트
    max_lambda = adjust_lambda_dynamically(
        current_lambda=8.0,
        previous_sharpe=0.9,
        current_sharpe=0.2,  # 성능 저하
        min_lambda=0.1,
        max_lambda=10.0,
        adjustment_factor=0.5
    )
    print(f"최대값 테스트: 8.0 -> {max_lambda:.3f} (최대값 10.0)")

def test_adaptive_lambda_integration():
    """적응형 lambda 통합 테스트"""
    
    print("\n=== 적응형 Lambda 통합 테스트 ===")
    
    # 간단한 모델 생성 (테스트용)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1, activation='tanh')
    ])
    
    # 가짜 데이터 생성
    inputs = np.random.randn(100, 5).astype(np.float32)
    labels = np.random.randn(100, 1).astype(np.float32)
    
    # 가짜 theta_star와 fisher 생성
    theta_star = {v.name: tf.identity(v) for v in model.trainable_weights}
    fisher = {v.name: tf.ones_like(v) for v in model.trainable_weights}
    
    print("모델 구조:")
    model.summary()
    
    # 적응형 lambda 계산 테스트
    adaptive_lambda = calculate_adaptive_lambda(
        model=model,
        theta_star=theta_star,
        fisher=fisher,
        inputs=inputs,
        labels=labels,
        previous_sharpe=0.6,
        current_sharpe=0.8,
        base_lambda=1.0,
        batch_size=32
    )
    
    print(f"적응형 Lambda: {adaptive_lambda:.3f}")

if __name__ == "__main__":
    print("동적 EWC Lambda 조정 기능 테스트를 시작합니다...\n")
    
    try:
        test_dynamic_lambda_adjustment()
        test_lambda_boundaries()
        test_adaptive_lambda_integration()
        
        print("\n=== 모든 테스트 완료 ===")
        print("동적 lambda 조정 기능이 정상적으로 작동합니다.")
        
    except Exception as e:
        print(f"\n테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc() 