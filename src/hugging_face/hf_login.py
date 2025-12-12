"""
Script para hacer login en Hugging Face
"""
from huggingface_hub import login, HfApi
import getpass
import os

def hf_login():
    """
    Login interactivo en Hugging Face
    """
    print("="*60)
    print("LOGIN EN HUGGING FACE")
    print("="*60)
    
    # Verificar si ya hay un token guardado
    try:
        api = HfApi()
        whoami = api.whoami()
        print(f"\n✓ Ya estás logueado como: {whoami['name']}")
        response = input("¿Quieres usar el token existente? (y/n): ")
        if response.lower() == 'y':
            print("✓ Usando token existente")
            return True
    except Exception:
        pass  # No hay token guardado
    
    print("\nPara obtener tu token:")
    print("1. Ve a https://huggingface.co/settings/tokens")
    print("2. Crea un nuevo token con permisos de 'Write'")
    print("3. Copia el token")
    print()
    
    # Solicitar token
    print("Pega tu token de Hugging Face y presiona Enter:")
    token = input().strip()
    
    if not token:
        print("✗ No se ingresó ningún token")
        return False
    
    try:
        # Hacer login
        login(token=token.strip(), add_to_git_credential=True)
        
        # Verificar login
        api = HfApi()
        whoami = api.whoami()
        
        print(f"\n✓ Login exitoso como: {whoami['name']}")
        print("✓ Token guardado para futuros usos")
        return True
    except Exception as e:
        print(f"\n✗ Error en el login: {e}")
        print("\nVerifica que:")
        print("- El token sea válido")
        print("- Tenga permisos de 'Write'")
        print("- No haya espacios al inicio o final")
        return False

def check_login():
    """
    Verifica si ya estás logueado
    """
    try:
        api = HfApi()
        whoami = api.whoami()
        print(f"✓ Ya estás logueado como: {whoami['name']}")
        return True
    except Exception:
        print("✗ No estás logueado")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_login()
    else:
        hf_login()