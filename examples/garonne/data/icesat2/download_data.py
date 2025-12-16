import earthaccess

# 1. Authentification à votre compte NASA Earthdata Login
# (Il vous demandera vos identifiants de manière interactive si ce n'est pas déjà configuré)
auth = earthaccess.login(strategy="interactive")

# 2. Recherche des granules du produit ATL22
# Vous pouvez spécifier une boîte englobante [lon_min, lat_min, lon_max, lat_max]
results = earthaccess.search_data(
    short_name='ATL22',
    bounding_box=(-64, -66, -60, -65),
    temporal=('2024-01-01', '2024-06-01')
)

print(f"Nombre de fichiers ATL22 trouvés : {len(results)}")

# 3. Téléchargement des fichiers HDF5 dans le dossier de votre choix
downloaded_files = earthaccess.download(results, local_path='./2019-2024')