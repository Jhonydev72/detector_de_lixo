-- 0. Habilitar a extensão espacial do PostGIS no banco de dados
CREATE EXTENSION IF NOT EXISTS postgis;

-- 1. Tabela Voo (O Evento Ampliado)
-- Agora possui o classificador 'tipo_voo' para rotear os dados corretamente.
CREATE TABLE Voo (
    id_voo SERIAL PRIMARY KEY,
    data_voo TIMESTAMP NOT NULL, -- Garante o referencial cronológico
    tipo_voo VARCHAR(20) NOT NULL CHECK (tipo_voo IN ('INSPECAO', 'MAPEAMENTO_2D'))
);

-- 2. Tabela Mapeamento_Ambiente (Novo Módulo 2D)
-- Destinada aos voos de mapeamento visual sem IA, mantendo o histórico limpo.
CREATE TABLE Mapeamento_Ambiente (
    id_mapeamento SERIAL PRIMARY KEY,
    id_voo INT NOT NULL,
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    caminho_imagem VARCHAR(255) NOT NULL,
    CONSTRAINT fk_voo_mapeamento 
        FOREIGN KEY (id_voo) 
        REFERENCES Voo(id_voo) 
        ON DELETE RESTRICT -- Impede apagar o voo se houver imagens vinculadas
);

-- 3. Tabela Deteccao (O Repositório Central de Infrações)
-- Permanece estritamente focada no lixo, armazenando o limiar de confiança do YOLOv8.
CREATE TABLE Deteccao (
    id_deteccao SERIAL PRIMARY KEY,
    id_voo INT NOT NULL,
    latitude DECIMAL(10, 8) NOT NULL, -- Restrição vital para precisão em centímetros
    longitude DECIMAL(11, 8) NOT NULL, -- Restrição vital para precisão em centímetros[cite: 1]
    confianca_ia DECIMAL(3, 2) NOT NULL CHECK (confianca_ia >= 0.00 AND confianca_ia <= 1.00), -- Domínio da rede neural[cite: 1]
    caminho_imagem VARCHAR(255) NOT NULL, -- Ponteiro para o Data Lake[cite: 1]
    CONSTRAINT fk_voo_deteccao 
        FOREIGN KEY (id_voo) 
        REFERENCES Voo(id_voo) 
        ON DELETE RESTRICT -- Regra de Restrição para manter a integridade referencial[cite: 1]
);

-- 4. Tabela Recorrencia (A Inteligência Relacional)
-- Tabela de junção N:N que mapeia instâncias históricas das detecções[cite: 1].
CREATE TABLE Recorrencia (
    id_recorrencia SERIAL PRIMARY KEY,
    id_deteccao_hoje INT NOT NULL,
    id_deteccao_passado INT NOT NULL,
    CONSTRAINT fk_deteccao_atual 
        FOREIGN KEY (id_deteccao_hoje) 
        REFERENCES Deteccao(id_deteccao) 
        ON DELETE RESTRICT,
    CONSTRAINT fk_deteccao_passado 
        FOREIGN KEY (id_deteccao_passado) 
        REFERENCES Deteccao(id_deteccao) 
        ON DELETE RESTRICT,
    -- Impede que a mesma detecção seja cruzada com ela mesma
    CONSTRAINT chk_deteccoes_diferentes 
        CHECK (id_deteccao_hoje != id_deteccao_passado) 
);