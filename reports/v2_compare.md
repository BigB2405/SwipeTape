# v2 comparison (validation; test split not opened)

| model | r50 | r50_mixed | genre_agree | kids_rate | n_genres |
|---|---|---|---|---|---|
| M5 (live) alpha=2 leaf=10 | 0.1392 | 0.0945 | 0.1935 | 0.0226 | 8.9714 |
| M5 + pop 0.05 | 0.1366 | 0.0889 | 0.1837 | 0.0147 | 9.6589 |
| Tags alpha=0.1 pop=0.0 | 0.1393 | 0.1066 | 0.2823 | 0.0225 | 8.2826 |
| Tags alpha=0.1 pop=0.05 | 0.1326 | 0.0924 | 0.2424 | 0.0156 | 9.2852 |
| Tags alpha=0.2 pop=0.0 | 0.232 | 0.1719 | 0.5704 | 0.0138 | 6.1823 |
| Tags alpha=0.2 pop=0.05 | 0.2194 | 0.1622 | 0.5161 | 0.0085 | 7.2461 |
| Tags alpha=0.3 pop=0.0 | 0.3396 | 0.2619 | 0.8272 | 0.0072 | 3.875 |
| Tags alpha=0.3 pop=0.05 | 0.3338 | 0.2502 | 0.8042 | 0.0036 | 4.3424 |
| Tags alpha=0.5 pop=0.0 | 0.5049 | 0.3921 | 0.9857 | 0.0006 | 2.1146 |
| Tags alpha=0.5 pop=0.05 | 0.5037 | 0.3915 | 0.9846 | 0.0004 | 2.2083 |
| Tags alpha=1.0 pop=0.0 | 0.5678 | 0.4337 | 0.9996 | 0.0 | 1.5312 |
| Tags alpha=1.0 pop=0.05 | 0.5696 | 0.4356 | 0.9996 | 0.0 | 1.5508 |

Best tag model by R@50: **Tags alpha=1.0 pop=0.05**

## Spot checks (full catalog)

**Gangsta's Paradise**
- M5 (live) alpha=2 leaf=10: ABC Boo – Super Simple Songs; Tera Hua Na Mai Kabhi VIP – Swattrex; Karolina – Kum; Só de Olhar – RDN; Yaariyan Lofi Flip – Ghauri; Meu Querubim – RDN
- Tags alpha=1.0 pop=0.05: God's Plan – Drake; Sarkar – Jaura Phagwara; I Wanna Fuck You – Snoop Dogg; Carro Bicho – Tz da Coronel; WITHOUT YOU – The Kid LAROI; MAJHE AALE – AP Dhillon

**Never Enough**
- M5 (live) alpha=2 leaf=10: Look at Me – Kelli O'Hara; Vivo Per Lei – Musical Tenors; Vuelie/太陽よ 輝いて – SHIKI THEATRE COMP; Heart Full of Praise – Phil Wickham; Reunimos-nos Aqui – Kemuel; En helt ny verden – Lasse Vermeli
- Tags alpha=1.0 pop=0.05: A Million Dreams – Ziv Zaifman; For Forever - From The “Dear – Ben Platt; Come to Your Senses – Alexandra Shipp; Real Life – Robin de Jesus; Requiem – Laura Dreyfuss; Fight for Me – Barrett Wilbert We

**Unholy (feat. Kim Petras)**
- M5 (live) alpha=2 leaf=10: I Don’t Wanna Live Forever ( – ZAYN; I Know You (feat. Bastille)  – Craig David; Hypocrite – Jok'air; Lanet – Arda Gezer; Skin do Kof – Yung Buda; Moving On – Lil Peep
- Tags alpha=1.0 pop=0.05: I Don’t Wanna Live Forever ( – ZAYN; New Love - Paraone Remix – Dua Lipa; Havana (feat. Young Thug) – Camila Cabello; Under The Influence – Chris Brown; You Right – Doja Cat; Need to Know – Doja Cat
