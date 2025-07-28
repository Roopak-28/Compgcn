def build_fb2wikidata_map(rdf_filename):
    fb2wikidata = {}
    with open(rdf_filename, 'r', encoding='utf-8') as f:
        for line in f:
            if '<http://www.w3.org/2002/07/owl#sameAs' in line:
                parts = line.strip().split()
                if len(parts) >= 3:
                    wd_uri = parts[2].strip('<>.')
                    fb_uri = parts[0].strip('<>')
                    if fb_uri.startswith('http://rdf.freebase.com/ns/'):
                        fb_id = fb_uri.split('/')[-1]        # e.g., m.02zyy4
                        fb_id_dot = fb_id.replace('/', '.')  # in case it has slashes
                        fb2wikidata[fb_id] = wd_uri.split('/')[-1]
                        fb2wikidata[fb_id_dot] = wd_uri.split('/')[-1]
    return fb2wikidata

rdf_file = '/Users/roopakkrishna/Downloads/fb2w.nt'
fb2wikidata = build_fb2wikidata_map(rdf_file)

def freebase_to_wikidata(fb_label, mapping):
    # Accepts either 'm.02zyy4', '/m.02zyy4', 'm/02zyy4', '/m/02zyy4'
    fb_label = fb_label.lstrip('/')       # remove leading slash if present
    if fb_label in mapping:
        return mapping[fb_label]
    fb_label_slash = fb_label.replace('.', '/')
    if fb_label_slash in mapping:
        return mapping[fb_label_slash]
    fb_label_dot = fb_label.replace('/', '.')
    if fb_label_dot in mapping:
        return mapping[fb_label_dot]
    return "No Wikidata ID found."

user_input = input("Enter the Freebase label (e.g., /m.02zyy4 or m/02zyy4): ").strip()
wikidata_id = freebase_to_wikidata(user_input, fb2wikidata)
print(f"Freebase {user_input} → Wikidata {wikidata_id}")
