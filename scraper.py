import requests
from bs4 import BeautifulSoup
import re
import urllib.parse
from urllib.parse import urljoin
import json

def scrape_audio_tags(url):
    """
    Scrape les balises audio depuis HowToPronounce.com
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        # Requête GET avec headers
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parser le HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Méthode 1: Chercher toutes les balises audio
        audio_tags = soup.find_all('audio')
        print(f"Balises audio trouvées: {len(audio_tags)}")
        
        audio_sources = []
        
        for i, audio in enumerate(audio_tags):
            print(f"\n--- Audio {i+1} ---")
            print(f"Balise complète: {audio}")
            
            # Extraire les attributs
            if audio.get('src'):
                src = audio.get('src')
                full_url = urljoin(url, src)
                audio_sources.append({
                    'type': 'direct_src',
                    'url': full_url,
                    'original_src': src
                })
                print(f"Source directe: {full_url}")
            
            # Chercher les balises source enfants
            sources = audio.find_all('source')
            for j, source in enumerate(sources):
                if source.get('src'):
                    src = source.get('src')
                    full_url = urljoin(url, src)
                    audio_sources.append({
                        'type': 'source_tag',
                        'url': full_url,
                        'original_src': src,
                        'mime_type': source.get('type', 'unknown')
                    })
                    print(f"Source {j+1}: {full_url} (Type: {source.get('type', 'unknown')})")
        
        # Méthode 2: Chercher dans le JavaScript/JSON embarqué
        print("\n--- Recherche dans les scripts ---")
        scripts = soup.find_all('script')
        
        for script in scripts:
            if script.string:
                # Chercher des URLs audio dans le JavaScript
                audio_urls = re.findall(r'["\']([^"\']*\.(?:mp3|wav|ogg|m4a)[^"\']*)["\']', script.string, re.IGNORECASE)
                for audio_url in audio_urls:
                    full_url = urljoin(url, audio_url)
                    audio_sources.append({
                        'type': 'js_embedded',
                        'url': full_url,
                        'original_src': audio_url
                    })
                    print(f"Audio trouvé dans JS: {full_url}")
                
                # Chercher des objets JSON contenant des URLs audio
                json_matches = re.findall(r'\{[^}]*["\'](?:audio|sound|pronunciation)["\'][^}]*\}', script.string, re.IGNORECASE)
                for match in json_matches:
                    print(f"JSON potentiel: {match}")
        
        # Méthode 3: Chercher des attributs data-* contenant des URLs audio
        print("\n--- Recherche dans les attributs data-* ---")
        elements_with_data = soup.find_all(attrs={"data-audio": True})
        elements_with_data.extend(soup.find_all(attrs={"data-src": True}))
        elements_with_data.extend(soup.find_all(attrs={"data-sound": True}))
        
        for element in elements_with_data:
            for attr, value in element.attrs.items():
                if attr.startswith('data-') and isinstance(value, str):
                    if any(ext in value.lower() for ext in ['.mp3', '.wav', '.ogg', '.m4a']):
                        full_url = urljoin(url, value)
                        audio_sources.append({
                            'type': f'data_attribute_{attr}',
                            'url': full_url,
                            'original_src': value,
                            'element': str(element)[:200] + "..."
                        })
                        print(f"Audio dans {attr}: {full_url}")
        
        # Méthode 4: Chercher tous les liens vers des fichiers audio
        print("\n--- Recherche de tous les liens audio ---")
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            href = link.get('href')
            if any(ext in href.lower() for ext in ['.mp3', '.wav', '.ogg', '.m4a']):
                full_url = urljoin(url, href)
                audio_sources.append({
                    'type': 'link',
                    'url': full_url,
                    'original_src': href,
                    'text': link.get_text(strip=True)
                })
                print(f"Lien audio: {full_url} ({link.get_text(strip=True)})")
        
        # Résumé
        print(f"\n=== RÉSUMÉ ===")
        print(f"Total d'URLs audio trouvées: {len(audio_sources)}")
        
        # Supprimer les doublons
        unique_urls = {}
        for source in audio_sources:
            url_key = source['url']
            if url_key not in unique_urls:
                unique_urls[url_key] = source
        
        print(f"URLs uniques: {len(unique_urls)}")
        
        for i, (url, info) in enumerate(unique_urls.items(), 1):
            print(f"{i}. {url} (Type: {info['type']})")
        
        return list(unique_urls.values())
        
    except requests.RequestException as e:
        print(f"Erreur de requête: {e}")
        return []
    except Exception as e:
        print(f"Erreur: {e}")
        return []

def download_audio_file(audio_url, filename=None):
    """
    Télécharge un fichier audio
    """
    if not filename:
        filename = audio_url.split('/')[-1]
        if not any(ext in filename.lower() for ext in ['.mp3', '.wav', '.ogg', '.m4a']):
            filename += '.mp3'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(audio_url, headers=headers, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Fichier téléchargé: {filename}")
        return True
    except Exception as e:
        print(f"Erreur lors du téléchargement: {e}")
        return False

# Utilisation
if __name__ == "__main__":
    url = "https://fr.howtopronounce.com/german/drau%C3%9Fen"
    
    print(f"Scraping de: {url}")
    audio_sources = scrape_audio_tags(url)
    
    if audio_sources:
        print(f"\n{len(audio_sources)} sources audio trouvées!")
        
        # Optionnel: télécharger le premier fichier audio trouvé
        choice = input("\nVoulez-vous télécharger le premier fichier audio? (y/n): ")
        if choice.lower() == 'y' and audio_sources:
            first_audio = audio_sources[0]
            download_audio_file(first_audio['url'], 'mich_pronunciation.mp3')
    else:
        print("Aucune source audio trouvée.")