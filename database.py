#!/usr/bin/env python3
import mysql.connector
from mysql.connector import Error
from config import DB_CONFIG
import json

class Database:
    def __init__(self):
        self.connection = None
        self.connect()
        self.setup_database()

    def connect(self):
        try:
            self.connection = mysql.connector.connect(**DB_CONFIG)
            print("Connexion à MySQL réussie")
        except Error as e:
            print(f"Erreur lors de la connexion à MySQL: {e}")
            raise

    def setup_database(self):
        try:
            cursor = self.connection.cursor()
            
            # Créer la table videos si elle n'existe pas
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    youtube_id VARCHAR(20) NOT NULL,
                    audio_path VARCHAR(255) NOT NULL,
                    json_data JSON,
                    status ENUM('pending', 'processing', 'completed', 'error') DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
            """)
            
            self.connection.commit()
            print("Base de données initialisée avec succès")
            
        except Error as e:
            print(f"Erreur lors de l'initialisation de la base de données: {e}")
            raise
        finally:
            cursor.close()

    def add_video(self, youtube_id, audio_path):
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO videos (youtube_id, audio_path)
                VALUES (%s, %s)
            """
            cursor.execute(query, (youtube_id, audio_path))
            self.connection.commit()
            return cursor.lastrowid
        except Error as e:
            print(f"Erreur lors de l'ajout de la vidéo: {e}")
            raise
        finally:
            cursor.close()

    def update_video_status(self, video_id, status):
        try:
            cursor = self.connection.cursor()
            query = """
                UPDATE videos
                SET status = %s
                WHERE id = %s
            """
            cursor.execute(query, (status, video_id))
            self.connection.commit()
        except Error as e:
            print(f"Erreur lors de la mise à jour du statut: {e}")
            raise
        finally:
            cursor.close()

    def update_video_json(self, video_id, json_data):
        try:
            cursor = self.connection.cursor()
            query = """
                UPDATE videos
                SET json_data = %s, status = 'completed'
                WHERE id = %s
            """
            cursor.execute(query, (json.dumps(json_data), video_id))
            self.connection.commit()
        except Error as e:
            print(f"Erreur lors de la mise à jour des données JSON: {e}")
            raise
        finally:
            cursor.close()

    def get_video(self, video_id):
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
                SELECT *
                FROM videos
                WHERE id = %s
            """
            cursor.execute(query, (video_id,))
            return cursor.fetchone()
        except Error as e:
            print(f"Erreur lors de la récupération de la vidéo: {e}")
            raise
        finally:
            cursor.close()

    def get_all_videos(self):
        """Récupère toutes les vidéos"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
                SELECT *
                FROM videos
                ORDER BY created_at DESC
            """
            cursor.execute(query)
            return cursor.fetchall()
        except Error as e:
            print(f"Erreur lors de la récupération des vidéos: {e}")
            raise
        finally:
            cursor.close()

    def delete_video(self, video_id):
        """Supprime une vidéo"""
        try:
            cursor = self.connection.cursor()
            query = """
                DELETE FROM videos
                WHERE id = %s
            """
            cursor.execute(query, (video_id,))
            self.connection.commit()
        except Error as e:
            print(f"Erreur lors de la suppression de la vidéo: {e}")
            raise
        finally:
            cursor.close()

    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Connexion à MySQL fermée")
