# read from a list of artists and get albums and songs from musicbrainz
import argh
import urllib.request as req
import numpy as np
from xml.etree import ElementTree as ET

# songs will overlap in title:
def check_song_name(song_dict, song_name, idx):
    sn = song_name
    i = 1
    while sn in song_dict:
        i += 1
        sn = song_name + "_" + str(i)

    song_dict[sn] = idx
    return sn

# urls we need for queries:
url_base = 'https://musicbrainz.org/ws/2/artist/?query='
url_base_rec = 'https://musicbrainz.org/ws/2/recording?query=arid:'
offset_str = '&offset='
limit_str = '&limit=100'

# xpath stuff
sch     = '{http://musicbrainz.org/ns/mmd-2.0#}'
artls   = 'artist-list'
art     = 'artist'
recls   = 'recording-list'
recs    = 'recording'
rells   = 'release-list'
rel     = 'release'
schrecs = sch+recls+'/'+sch+recs
schrels = sch+rells+'/'+sch+rel
schart  = sch+artls+'/'+sch+art
schrecolist = sch+recls

@argh.arg("-s", "--songsmax", help="Max. number of songs per artist to find")
@argh.arg("-a", "--artistlist", help="Input file of artists")
@argh.arg("-o", "--outputinfo", help="Output file of information")
@argh.arg("-e", "--edgelist", help="Edge list file output")

def buildmusic(artistlist="data/artists.txt", outputinfo="data/music_info.txt", songsmax=100, edgelist="edges/music.edges"):
    fp = open(artistlist, 'r')
    fp2 = open(outputinfo, 'w')

    # build catalogs and edgelist for embedding
    dict_artists = open('dicts/dict_artists.txt', 'w')
    dict_albums = open('dicts/dict_albums.txt', 'w')
    dict_songs = open('dicts/dict_songs.txt', 'w')
    edge_lists = open(edgelist, 'w')

    album_dict = {}
    song_dict  = {}
    idx = 1

    for line in fp:
        words = line.split()
        artist_str = '\%20'.join(words)

        print("Getting music by ", artist_str)

        # let's download stuff:
        html = req.urlopen(url_base+artist_str).read()

        # getting the relevant artist is tricky; we just grab the first and hope it's right
        r = ET.fromstring(html)
        #artists = r.findall(schart)
        artist = r.find(schart)
        n_found = len(artist)

        '''
        # possible we found several artists, so disambiguate them:
        for artist in artists:
            # by exact name?
            artist_name = artist.find(sch + 'name')
            if artist_name.text == artist_str:
                break

            diso = artist.find(sch+'disambiguation')
            if diso is not None:
                dis = diso.text
                if dis.find('rapper')!=-1 or dis.find('Rapper')!=-1:
                    break
            '''

        if n_found > 0:
            # found an artist:
            artist_name = artist.find(sch + 'name')
            artist_id = artist.attrib['id']
            artist_idx = idx
            idx += 1
            dict_artists.write(artist_name.text + "\t" + str(artist_idx) + "\n")

            # implicit forest embedding
            edge_lists.write('0' + "\t" + str(artist_idx) + "\t" + '10'  + "\n") 

            # now let's get their songs:
            tot_hits = np.Inf
            song_offset = 0

            while song_offset < songsmax and song_offset < tot_hits:
                # no offset for first call:
                if song_offset == 0:
                    html2 = req.urlopen(url_base_rec+artist_id+limit_str).read()
                    rec_r = ET.fromstring(html2)

                    # get the number of hits:
                    rec_list = rec_r.find(schrecolist)
                    tot_hits = int(rec_list.attrib['count'])
                else:
                    html2 = req.urlopen(url_base_rec+artist_id+limit_str+offset_str+str(song_offset)).read()
                    rec_r = ET.fromstring(html2)

                song_offset += 100

                # get their songs:
                recordings = rec_r.findall(schrecs)

                for record in recordings:
                    song_name = record.find(sch+'title')
                    sn = song_name.text

                    # try to find the albums corresponding to each song:
                    album = record.find(schrels)
                    if album:
                        song_idx = idx
                        sn = check_song_name(song_dict, sn, song_idx)
                        dict_songs.write(sn + "\t" + str(song_idx)  + "\n")
                        idx += 1

                        album_name = album.find(sch+'title')
                        if album_name.text not in album_dict:
                            album_dict[album_name.text] = idx
                            idx += 1
                            dict_albums.write(album_name.text + "\t" + str(album_dict[album_name.text]) + "\n")
                            edge_lists.write(str(artist_idx) + "\t" + str(album_dict[album_name.text]) + "\t" +  '1' + "\n")

                        edge_lists.write(str(album_dict[album_name.text]) + "\t" + str(song_idx) + "\t" + '2'  + "\n") 

                        # write everything in format ARTIST TAB ALBUM TAB SONG for future reference
                        fp2.write(artist_name.text + "\t" + album_name.text + "\t" + sn + "\n")   

    fp.close()
    fp2.close()
    dict_artists.close()
    dict_albums.close()
    dict_songs.close()
    edge_lists.close()

    return

if __name__ == '__main__':
    _parser = argh.ArghParser()
    _parser.add_commands([buildmusic])
    _parser.dispatch()
