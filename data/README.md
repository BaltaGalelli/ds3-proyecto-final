# Datos

## Banking77

| | |
|---|---|
| **Qué es** | 13.083 consultas reales de clientes de un banco, en inglés, etiquetadas con 77 motivos de consulta |
| **Partición oficial** | 10.003 de train (entre 35 y 187 por motivo) y 3.080 de test (40 por motivo) |
| **Fuente** | [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data) — también en [Hugging Face](https://huggingface.co/datasets/PolyAI/banking77) |
| **Licencia** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): uso libre citando a los autores |
| **Paper** | Casanueva, I., Temčinas, T., Gerz, D., Henderson, M. y Vulić, I. (2020). *Efficient Intent Detection with Dual Sentence Encoders*. Proceedings of the 2nd Workshop on NLP for ConvAI, ACL 2020 |

## Cómo se obtiene

No hay que descargar nada a mano: la primera vez que se corre el pipeline, `src/pipeline/datos.py` baja `train.csv` y `test.csv` a `data/banking77/`.

- La URL está **fijada a un commit** del repositorio oficial (`57ec275`), así que siempre se descarga la misma versión.
- Cada archivo se verifica con su **hash SHA-256**; si no coincide, se borra y el pipeline se detiene en lugar de entrenar con datos distintos.

| Archivo | SHA-256 |
|---|---|
| `train.csv` | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` |
| `test.csv` | `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` |

`data/banking77/` está en el `.gitignore`: los datos se documentan acá en lugar de subirlos al repositorio.
