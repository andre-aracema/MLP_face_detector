import numpy as np
import os
import sys
import numpy.random as npr


# Cria um novo conjunto de dados (v3) que é balanceado: 
# 50% não-faces fáceis (v1) e 50% não-faces difíceis (v2)
def create_mixed_dataset(data_path_v1, data_path_v2, save_path_base):
    print("Iniciando Processo de Mixagem 50/50 (v3)")

    try:
        # Carregar faces (base)
        face_path = f"{data_path_v1}_face.npy"
        print(f"Carregando faces de {face_path}")
        base_faces = np.load(face_path)
        num_faces = len(base_faces)
        print(f"Encontradas {num_faces} faces. Esta será nossa meta.")

        # Carregar não-faces fáceis
        easy_non_face_path = f"{data_path_v1}_non_face.npy"
        print(f"Carregando não-faces fáceis de {easy_non_face_path}")
        easy_non_faces = np.load(easy_non_face_path)
        num_easy = len(easy_non_faces)
        print(f"Encontradas {num_easy} não-faces fáceis.")

        # Carregar não-faces difíceis (hard negatives)
        hard_non_face_path = f"{data_path_v2}_non_face.npy"
        print(f"Carregando não-faces difíceis de {hard_non_face_path}")
        hard_non_faces = np.load(hard_non_face_path)
        num_hard = len(hard_non_faces)
        print(f"Encontradas {num_hard} não-faces difíceis.")

    except FileNotFoundError as e:
        print(f"\nErro: Arquivo .npy não encontrado.")
        print(f"Detalhe: {e}")
        print("Verifique se 'preprocess' e 'bootstrap' já foram executados.")
        sys.exit(1)

    if num_easy == 0 or num_hard == 0:
        print("Erro: Arquivos de não-faces fáceis ou difíceis estão vazios. Abortando.")
        sys.exit(1)

    # Definir metas 50/50
    target_total = num_faces
    target_hard = target_total // 2
    target_easy = target_total - target_hard 

    print(f"\nMeta de Amostragem: {target_total} não-faces, sendo:")
    print(f"  - {target_easy} fáceis (v1)")
    print(f"  - {target_hard} difíceis (v2)")

    # Lidar com escassez (ex: não temos 'target_hard' amostras)
    final_easy_samples = []
    final_hard_samples = []

    if num_hard < target_hard:
        # Não temos 'hard' suficientes. Pegar todos e preencher com 'easy'.
        print(f"AVISO: {num_hard} 'hard' é menor que a meta {target_hard}.")
        print("       Usando todos os 'hard' e preenchendo o restante com 'easy'.")
        
        final_hard_samples = hard_non_faces
        
        remaining_needed = target_total - len(final_hard_samples)
        if remaining_needed > num_easy:
             print(f"Erro: Não há amostras 'easy' ({num_easy}) suficientes para preencher {remaining_needed}.")
             sys.exit(1)
             
        easy_indices = npr.choice(num_easy, remaining_needed, replace=False)
        final_easy_samples = easy_non_faces[easy_indices]

    elif num_easy < target_easy:
         # Não temos 'easy' suficientes. Pegar todos e preencher com 'hard'.
        print(f"AVISO: {num_easy} 'easy' é menor que a meta {target_easy}.")
        print("       Usando todos os 'easy' e preenchendo o restante com 'hard'.")

        final_easy_samples = easy_non_faces

        remaining_needed = target_total - len(final_easy_samples)
        if remaining_needed > num_hard:
             print(f"Erro: Não há amostras 'hard' ({num_hard}) suficientes para preencher {remaining_needed}.")
             sys.exit(1)

        hard_indices = npr.choice(num_hard, remaining_needed, replace=False)
        final_hard_samples = hard_non_faces[hard_indices]
        
    else:
        # Caso Ideal: Temos amostras suficientes de ambos
        print("Ambos os conjuntos têm amostras suficientes. Realizando amostragem 50/50.")
        easy_indices = npr.choice(num_easy, target_easy, replace=False)
        final_easy_samples = easy_non_faces[easy_indices]
        
        hard_indices = npr.choice(num_hard, target_hard, replace=False)
        final_hard_samples = hard_non_faces[hard_indices]

    # Combinar e Salvar
    print(f"\nAmostragem final de não-faces:")
    print(f"  - {len(final_easy_samples)} fáceis")
    print(f"  - {len(final_hard_samples)} difíceis")
    print(f"  - Total: {len(final_easy_samples) + len(final_hard_samples)}")

    final_non_faces = np.concatenate((final_easy_samples, final_hard_samples), axis=0)
    
    # Embaralhar o resultado final
    npr.shuffle(final_non_faces)
    
    final_faces = base_faces 

    # Salvar em novos arquivos
    os.makedirs(os.path.dirname(save_path_base), exist_ok=True)
    save_path_face = f"{save_path_base}_face.npy"
    save_path_non_face = f"{save_path_base}_non_face.npy"
    
    np.save(save_path_face, final_faces)
    np.save(save_path_non_face, final_non_faces)
    
    print(f"\nProcesso de Mixagem 50/50 Concluído")
    print(f"Dados v3 (mixados) salvos em:")
    print(f"  - {save_path_face} ({len(final_faces)} amostras)")
    print(f"  - {save_path_non_face} ({len(final_non_faces)} amostras)")
    print(f"\nPróximo passo: Treine o modelo 3 com 'python main.py train --config v3'")