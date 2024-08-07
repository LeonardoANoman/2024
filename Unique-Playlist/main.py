import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from requests.exceptions import ReadTimeout, HTTPError
import json
import os
from typing import Set, List, Dict, Tuple
from keys import client_id, client_secret

FOLLOWED_ARTISTS_FILE = 'followed_artists.json'
PLAYLIST_ID = '1C9TEGqEscmDADF5VK4uhI'
MAX_RETRIES = 3
DELAY = 5

def debug_print(message: str) -> None:
    print(f"[DEBUG] {message}")

def load_followed_artists() -> Set[str]:
    if os.path.exists(FOLLOWED_ARTISTS_FILE):
        with open(FOLLOWED_ARTISTS_FILE, 'r') as file:
            return set(json.load(file))
    return set()

def save_followed_artists(followed_artists: Set[str]) -> None:
    with open(FOLLOWED_ARTISTS_FILE, 'w') as file:
        json.dump(list(followed_artists), file)

def create_spotify_client() -> spotipy.Spotify:
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri='https://localhost:8080/callback',
        scope=['playlist-read-private', 'user-follow-modify', 'user-follow-read']
    ))

def fetch_playlist_tracks(sp: spotipy.Spotify, playlist_id: str) -> Tuple[Dict[str, List[str]], Set[str]]:
    artists_tracks = {}
    all_artists = set()
    playlist = sp.playlist_tracks(playlist_id, limit=100)
    
    while True:
        for item in playlist['items']:
            track = item['track']
            first_artist = track['artists'][0]['name']
            track_name = track['name']
            all_artists.add(first_artist)
            artists_tracks.setdefault(first_artist, []).append(track_name)

        debug_print(f"Processed {len(playlist['items'])} tracks")

        if not playlist['next']:
            break
        playlist = sp.next(playlist)

    return artists_tracks, all_artists

def get_artist_ids(sp: spotipy.Spotify, artists: Set[str]) -> Tuple[List[str], Dict[str, str]]:
    artist_ids = []
    artist_names_with_ids = {}
    for artist in artists:
        for _ in range(MAX_RETRIES):
            try:
                debug_print(f"Searching for artist: {artist}")
                results = sp.search(q='artist:' + artist, type='artist')
                if results['artists']['items']:
                    artist_id = results['artists']['items'][0]['id']
                    artist_ids.append(artist_id)
                    artist_names_with_ids[artist_id] = artist
                break
            except ReadTimeout:
                print(f"Timeout occurred for artist {artist}. Retrying...")
                time.sleep(DELAY)
        else:
            print(f"Failed to retrieve artist {artist} after {MAX_RETRIES} retries.")
    return artist_ids, artist_names_with_ids

def get_followed_artists_from_spotify(sp: spotipy.Spotify) -> Set[str]:
    followed_artists = set()
    results = sp.current_user_followed_artists(limit=50)
    while results:
        followed_artists.update(artist['id'] for artist in results['artists']['items'])
        if not results['artists']['next']:
            break
        results = sp.next(results['artists'])
    return followed_artists

def follow_artists_individually(sp: spotipy.Spotify, artist_ids: List[str], 
                                artist_names_with_ids: Dict[str, str], 
                                followed_artists: Set[str]) -> None:
    for artist_id in artist_ids:
        if artist_id not in followed_artists:
            try:
                sp.user_follow_artists([artist_id])
                print(f'Followed artist: {artist_names_with_ids[artist_id]}')
                followed_artists.add(artist_id)
                save_followed_artists(followed_artists)
            except HTTPError as e:
                print(f"HTTP error occurred for artist {artist_names_with_ids[artist_id]}: {e}")
        else:
            print(f"Already following artist: {artist_names_with_ids[artist_id]}")

def main():
    sp = create_spotify_client()
    followed_artists = load_followed_artists()

    debug_print("Fetching playlist tracks")
    artists_tracks, all_artists = fetch_playlist_tracks(sp, PLAYLIST_ID)

    repeated_artists = [artist for artist, tracks in artists_tracks.items() if len(tracks) > 1]

    if repeated_artists:
        for artist in repeated_artists:
            print(f'The artist {artist} is repeated in the playlist with the following tracks:')
            for track in artists_tracks[artist]:
                print(f'- {track}')
    else:
        print('All artists in the playlist are unique.')

    debug_print("Getting artist IDs")
    artist_ids, artist_names_with_ids = get_artist_ids(sp, all_artists)

    debug_print("Getting followed artists from Spotify")
    spotify_followed_artists = get_followed_artists_from_spotify(sp)
    all_followed_artists = followed_artists.union(spotify_followed_artists)

    debug_print("Following artists individually")
    follow_artists_individually(sp, artist_ids, artist_names_with_ids, all_followed_artists)

    save_followed_artists(all_followed_artists)

    print("Followed all artists in the playlist.")

if __name__ == "__main__":
    main()