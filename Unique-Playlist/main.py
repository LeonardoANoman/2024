import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from requests.exceptions import ReadTimeout, HTTPError
from keys import client_id, client_secret

def debug_print(message):
    print(f"[DEBUG] {message}")

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri='https://localhost:8080/callback',
                                               scope=['playlist-read-private', 'user-follow-modify', 'user-follow-read']))

playlist_id = '1C9TEGqEscmDADF5VK4uhI'

debug_print("Fetching initial playlist tracks")
playlist = sp.playlist_tracks(playlist_id, limit=100)

artists_tracks = {}
repeated_artists = []
all_artists = set()  

def fetch_playlist_tracks(playlist):
    while True:
        for track in playlist['items']:
            first_artist = track['track']['artists'][0]['name']
            track_name = track['track']['name']
            all_artists.add(first_artist)
            if first_artist in artists_tracks:
                repeated_artists.append(first_artist)
                artists_tracks[first_artist].append(track_name)
            else:
                artists_tracks[first_artist] = [track_name]

        debug_print(f"Processed {len(playlist['items'])} tracks")

        if playlist['next']:
            debug_print("Fetching next set of tracks")
            playlist = sp.next(playlist)
        else:
            break

def get_artist_ids(artists, max_retries=3, delay=5):
    artist_ids = []
    artist_names_with_ids = {}
    for artist in artists:
        retries = 0
        while retries < max_retries:
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
                retries += 1
                time.sleep(delay)
        else:
            print(f"Failed to retrieve artist {artist} after {max_retries} retries.")
    return artist_ids, artist_names_with_ids

def get_followed_artists():
    followed_artists = set()
    results = sp.current_user_followed_artists(limit=50)
    while results:
        for artist in results['artists']['items']:
            followed_artists.add(artist['id'])
        if results['artists']['next']:
            results = sp.next(results['artists'])
        else:
            break
    return followed_artists

def follow_artists_individually(artist_ids, artist_names_with_ids, followed_artists):
    for artist_id in artist_ids:
        if artist_id not in followed_artists:
            try:
                sp.user_follow_artists([artist_id])
                print(f'Followed artist: {artist_names_with_ids[artist_id]}')
            except HTTPError as e:
                print(f"HTTP error occurred for artist {artist_names_with_ids[artist_id]}: {e}")
        else:
            print(f"Already following artist: {artist_names_with_ids[artist_id]}")

debug_print("Fetching playlist tracks")
fetch_playlist_tracks(playlist)

repeated_artists = list(set(repeated_artists))

if repeated_artists:
    for artist in repeated_artists:
        print(f'The artist {artist} is repeated in the playlist with the following tracks:')
        for track in artists_tracks[artist]:
            print(f'- {track}')
else:
    print('All artists in the playlist are unique.')

debug_print("Getting artist IDs")
artist_ids, artist_names_with_ids = get_artist_ids(all_artists)

debug_print("Getting followed artists")
followed_artists = get_followed_artists()

debug_print("Following artists individually")
follow_artists_individually(artist_ids, artist_names_with_ids, followed_artists)

print("Followed all artists in the playlist.")
