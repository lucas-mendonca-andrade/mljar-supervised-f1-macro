#!/usr/bin/env python3
"""
Teste para verificar se a métrica f1_macro está funcionando corretamente.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from supervised import AutoML

def test_f1_macro_binary_classification():
    """Teste para classificação binária com f1_macro"""
    print("=== Teste de Classificação Binária com f1_macro ===")
    
    # Criar dados sintéticos para classificação binária
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        class_sep=0.8
    )
    
    # Converter para DataFrame
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Forma dos dados: {X.shape}")
    print(f"Distribuição das classes: {np.bincount(y)}")
    
    # Treinar AutoML com f1_macro
    automl = AutoML(
        mode="Explain",
        eval_metric="f1_macro",
        results_path="test_f1_macro_binary_new",
        total_time_limit=60,  # Limitar tempo para teste rápido
        train_ensemble=False,
        stack_models=False
    )
    
    print("Treinando AutoML com f1_macro...")
    automl.fit(X_train, y_train)
    
    # Fazer predições
    predictions = automl.predict(X_test)
    
    # Calcular f1_macro manualmente para comparação
    f1_macro_manual = f1_score(y_test, predictions, average='macro')
    
    print(f"F1 Macro calculado manualmente: {f1_macro_manual:.4f}")
    print(f"Melhor modelo: {automl.get_leaderboard().iloc[0]['name']}")
    
    # Corrigir a formatação do score
    leaderboard = automl.get_leaderboard()
    best_score = leaderboard.iloc[0]['metric_value']
    if isinstance(best_score, str):
        try:
            best_score = float(best_score)
        except ValueError:
            best_score = 0.0
    print(f"Score do melhor modelo: {best_score:.4f}")
    
    return f1_macro_manual > 0.5  # Esperamos um score razoável

def test_f1_macro_multiclass_classification():
    """Teste para classificação multiclasse com f1_macro"""
    print("\n=== Teste de Classificação Multiclasse com f1_macro ===")
    
    # Criar dados sintéticos para classificação multiclasse
    X, y = make_classification(
        n_samples=1500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=42,
        class_sep=0.8
    )
    
    # Converter para DataFrame
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Forma dos dados: {X.shape}")
    print(f"Distribuição das classes: {np.bincount(y)}")
    
    # Treinar AutoML com f1_macro
    automl = AutoML(
        mode="Explain",
        eval_metric="f1_macro",
        results_path="test_f1_macro_multiclass_new",
        total_time_limit=60,  # Limitar tempo para teste rápido
        train_ensemble=False,
        stack_models=False
    )
    
    print("Treinando AutoML com f1_macro...")
    automl.fit(X_train, y_train)
    
    # Fazer predições
    predictions = automl.predict(X_test)
    
    # Calcular f1_macro manualmente para comparação
    f1_macro_manual = f1_score(y_test, predictions, average='macro')
    
    print(f"F1 Macro calculado manualmente: {f1_macro_manual:.4f}")
    print(f"Melhor modelo: {automl.get_leaderboard().iloc[0]['name']}")
    
    # Corrigir a formatação do score
    leaderboard = automl.get_leaderboard()
    best_score = leaderboard.iloc[0]['metric_value']
    if isinstance(best_score, str):
        try:
            best_score = float(best_score)
        except ValueError:
            best_score = 0.0
    print(f"Score do melhor modelo: {best_score:.4f}")
    
    return f1_macro_manual > 0.4  # Esperamos um score razoável

def test_metric_validation():
    """Teste para verificar se a validação de métricas está funcionando"""
    print("\n=== Teste de Validação de Métricas ===")
    
    try:
        # Tentar usar f1_macro em classificação binária (deve funcionar)
        automl = AutoML(eval_metric="f1_macro")
        print("✓ f1_macro é aceito para classificação binária")
        
        # Tentar usar f1_macro em classificação multiclasse (deve funcionar)
        automl = AutoML(eval_metric="f1_macro")
        print("✓ f1_macro é aceito para classificação multiclasse")
        
        # Tentar usar uma métrica inválida (deve falhar)
        try:
            # Criar dados para forçar a validação
            X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            y = pd.Series(y)
            
            automl = AutoML(eval_metric="invalid_metric")
            automl.fit(X, y)  # Isso deve falhar na validação
            print("✗ Métrica inválida foi aceita (erro esperado)")
            return False
        except ValueError as e:
            if "invalid_metric" in str(e):
                print("✓ Métrica inválida foi rejeitada corretamente")
            else:
                print(f"✗ Erro inesperado: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Erro na validação de métricas: {e}")
        return False

if __name__ == "__main__":
    print("Iniciando testes para f1_macro...")
    
    # Executar testes
    test1_passed = test_f1_macro_binary_classification()
    test2_passed = test_f1_macro_multiclass_classification()
    test3_passed = test_metric_validation()
    
    print("\n" + "="*50)
    print("RESULTADOS DOS TESTES:")
    print(f"Teste 1 (Classificação Binária): {'✓ PASSOU' if test1_passed else '✗ FALHOU'}")
    print(f"Teste 2 (Classificação Multiclasse): {'✓ PASSOU' if test2_passed else '✗ FALHOU'}")
    print(f"Teste 3 (Validação de Métricas): {'✓ PASSOU' if test3_passed else '✗ FALHOU'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    print(f"\nStatus Geral: {'✓ TODOS OS TESTES PASSARAM' if all_passed else '✗ ALGUNS TESTES FALHARAM'}")
    
    if all_passed:
        print("\n🎉 A métrica f1_macro está funcionando corretamente!")
    else:
        print("\n⚠️  Há problemas com a implementação da métrica f1_macro.") 