import spotipy
from spotipy.oauth2 import SpotifyOAuth
from keys import client_id, client_secret

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri='https://localhost:8080/callback',
                                               scope='playlist-read-private'))

playlist_id = '1C9TEGqEscmDADF5VK4uhI'

playlist = sp.playlist_tracks(playlist_id, limit=100)

artists_tracks = {}
repeated_artists = []


def fetch_playlist_tracks(playlist):
    while True:
        for track in playlist['items']:
            first_artist = track['track']['artists'][0]['name']
            track_name = track['track']['name']
            if first_artist in artists_tracks:
                repeated_artists.append(first_artist)
                artists_tracks[first_artist].append(track_name)
            else:
                artists_tracks[first_artist] = [track_name]

        if playlist['next']:
            playlist = sp.next(playlist)
        else:
            break


fetch_playlist_tracks(playlist)

repeated_artists = list(set(repeated_artists))

if repeated_artists:
    for artist in repeated_artists:
        print(f'The artist {artist} is repeated in the playlist with the following tracks:')
        for track in artists_tracks[artist]:
            print(f'- {track}')
else:
    print('All artists in the playlist are unique.')
