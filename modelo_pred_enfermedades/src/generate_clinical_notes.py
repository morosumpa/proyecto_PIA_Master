import pandas as pd
import random
import os

# Plantillas para notas clínicas simuladas
clinical_note_templates = [
    "Paciente con antecedentes de hipertensión y colesterol alto.",
    "Paciente fumador, presenta síntomas leves de fatiga y dificultad para caminar.",
    "Paciente con índice de masa corporal elevado y actividad física limitada.",
    "Paciente con historial de ataques cardíacos y consumo de alcohol ocasional.",
    "Paciente con buen estado general, sin antecedentes de enfermedades crónicas.",
    "Paciente reporta dolor muscular y problemas para mantener una dieta saludable.",
    "Paciente con alta presión arterial y antecedentes familiares de diabetes.",
    "Paciente con historial de insuficiencia renal y hábitos de vida sedentarios.",
    "Paciente muestra signos de estrés y mala salud mental reciente.",
    "Paciente con diagnóstico reciente de diabetes y cambios en la actividad física."
]

def generate_clinical_notes(df):
    notes = []
    for _ in range(len(df)):
        note = random.choice(clinical_note_templates)
        notes.append(note)
    return notes

if __name__ == "__main__":
    filepath = os.path.join("..", "data", "diabetes_data.csv")
    df = pd.read_csv(filepath)

    df['Clinical_Notes'] = generate_clinical_notes(df)

    output_path = os.path.join("..", "data", "diabetes_data_with_notes.csv")
    df.to_csv(output_path, index=False)

    print(f"Archivo generado con notas clínicas simuladas: {output_path}")
