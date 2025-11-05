
import streamlit as st
import pandas as pd
import numpy as np
import os, io, time, datetime as dt
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine

st.set_page_config(page_title="AI Herbal Skincare Marketplace", page_icon="🪴", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = "sqlite:///skinmarket.db"
engine = create_engine(DB_PATH, echo=False, future=True)

# ---------------- DB ----------------
def init_db():
    with engine.begin() as c:
        c.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE, password TEXT, role TEXT
        )""")
        c.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS products(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, category TEXT, price REAL, stock INTEGER, rating REAL,
            ingredients TEXT, description TEXT, image_path TEXT, seller TEXT, skin_types TEXT
        )""")
        c.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS chats(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            buyer_email TEXT, seller TEXT, product_id INTEGER,
            sender TEXT, message TEXT, ts TEXT
        )""")
        c.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS orders(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            buyer_email TEXT, product_id INTEGER, qty INTEGER, total REAL, ts TEXT, payment_mode TEXT
        )""")
    # demo users
    with engine.begin() as c:
        for e,p,r in [("buyer@demo","buyer","buyer"),("seller@demo","seller","seller"),("admin@demo","admin","admin")]:
            c.exec_driver_sql("INSERT OR IGNORE INTO users(email,password,role) VALUES(?,?,?)",(e,p,r))
    # seed products (from CSV) if empty
    with engine.begin() as c:
        cnt = c.exec_driver_sql("SELECT COUNT(*) FROM products").scalar()
        if cnt==0:
            df = pd.read_csv(os.path.join(BASE_DIR,"data","products.csv"))
            for _,r in df.iterrows():
                c.exec_driver_sql("""
                INSERT INTO products(id,name,category,price,stock,rating,ingredients,description,image_path,seller,skin_types)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""", tuple(r.values))

def load_products():
    with engine.begin() as c:
        return pd.read_sql("SELECT * FROM products", c)

def next_product_id():
    with engine.begin() as c:
        val = c.exec_driver_sql("SELECT COALESCE(MAX(id),0)+1 FROM products").scalar()
        return int(val or 1)

def save_chat(buyer_email, seller, product_id, sender, message):
    with engine.begin() as c:
        c.exec_driver_sql("""
        INSERT INTO chats(buyer_email,seller,product_id,sender,message,ts)
        VALUES(?,?,?,?,?,?)""", (buyer_email,seller,int(product_id) if product_id else None,sender,message,dt.datetime.now().isoformat()))

def get_chat(buyer_email, seller, product_id=None, limit=200):
    q = "SELECT * FROM chats WHERE buyer_email=? AND seller=?"
    params=[buyer_email, seller]
    if product_id:
        q += " AND product_id=?"; params.append(int(product_id))
    q += " ORDER BY id ASC LIMIT ?"; params.append(limit)
    with engine.begin() as c:
        return pd.read_sql(q, c, params=tuple(params))

def place_order(buyer_email, product_id, qty, price, payment_mode="COD"):
    total = float(qty) * float(price)
    with engine.begin() as c:
        c.exec_driver_sql("""
        INSERT INTO orders(buyer_email,product_id,qty,total,ts,payment_mode)
        VALUES(?,?,?,?,?,?)""", (buyer_email,int(product_id),int(qty),total,dt.datetime.now().isoformat(),payment_mode))
        c.exec_driver_sql("UPDATE products SET stock = stock - ? WHERE id = ?", (int(qty), int(product_id)))
    return total

# ---------------- AI ----------------
@st.cache_data(show_spinner=False)
def build_vectors(df):
    text = (df["name"].fillna("")+" "+df["description"].fillna("")+" "+df["ingredients"].fillna("")+" "+df["category"].fillna("")+" "+df["skin_types"].fillna(""))
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words="english")
    X = vec.fit_transform(text)
    return vec, X

def recommend(df, vec, X, q="", skin="all", k=12):
    if not q:
        mask = df["skin_types"].str.contains(skin.split()[0], case=False, na=False) if skin and skin!="all" else np.ones(len(df), dtype=bool)
        return df[mask].sort_values(["rating","stock"], ascending=[False,False]).head(k)
    sims = cosine_similarity(vec.transform([q]), X)[0]
    ranked = df.iloc[np.argsort(-sims)]
    if skin and skin!="all":
        ranked = ranked[ranked["skin_types"].str.contains(skin.split()[0], case=False, na=False)]
    return ranked.head(k)

AVOID = {"oily":["coconut oil","shea"],"dry":["witch hazel","tea tree (high %)"],"sensitive":["fragrance","alcohol"],"acne-prone":["coconut oil","heavy butter"]}

def ingredient_advice(ingredients, skin):
    ing = [i.strip().lower() for i in str(ingredients).split(",")]
    bad = AVOID.get((skin or "").lower(), [])
    flagged = [i for i in ing if any(w in i for w in bad)]
    if flagged: tip = f"⚠️ For {skin}: consider avoiding → " + ", ".join(flagged)
    else: tip = f"✅ Ingredients look friendly for {skin or 'all'} skin."
    if "aloe" in str(ingredients).lower(): tip += " 👍 Aloe is soothing."
    if "niacinamide" in str(ingredients).lower(): tip += " ✨ Niacinamide helps with spots."
    return tip

def intent(m):
    m=m.lower()
    if any(k in m for k in ["price","cost","how much","rs","₹"]): return "price"
    if any(k in m for k in ["stock","available"]): return "stock"
    if any(k in m for k in ["ship","delivery","arrive","days"]): return "ship"
    if any(k in m for k in ["suit","skin","oily","dry","sensitive","acne"]): return "fit"
    if any(k in m for k in ["hi","hello","hey"]): return "hi"
    return "other"

def reply(kind, row=None, skin="all"):
    if kind=="price" and row is not None: return f"Price is ₹{row['price']:.0f}."
    if kind=="stock" and row is not None: return f"In stock: {int(row['stock'])}."
    if kind=="ship": return "Delivery: 3–5 days standard; 1–2 days express (metros)."
    if kind=="fit" and row is not None: return ingredient_advice(row["ingredients"], skin)
    if kind=="hi": return "Hello! Ask about price, stock, shipping or suitability."
    return "Could you share your goal (hydration / glow / acne)?"

# ---------------- Auth ----------------
def auth_box():
    st.sidebar.header("Account")
    if "user" not in st.session_state:
        with st.sidebar.form("login"):
            email = st.text_input("Email", "buyer@demo")
            pw = st.text_input("Password", "buyer", type="password")
            ok = st.form_submit_button("Sign in")
        if ok:
            role = "buyer" if email.startswith("buyer") else ("seller" if email.startswith("seller") else "admin")
            st.session_state.user={"email":email,"role":role}; st.sidebar.success(f"Signed in as {email} ({role})"); st.rerun()
        st.sidebar.caption("Demo: buyer@demo/buyer, seller@demo/seller, admin@demo/admin")
    else:
        u=st.session_state.user; st.sidebar.success(f"{u['email']} ({u['role']})")
        if st.sidebar.button("Sign out"): del st.session_state["user"]; st.rerun()

# ---------------- Seller utils (image upload) ----------------
def save_uploaded_image(file, suggested_name):
    # ensure images dir
    img_dir = os.path.join(BASE_DIR,"images")
    os.makedirs(img_dir, exist_ok=True)
    # safe filename
    base = suggested_name.replace(" ", "-").replace("/", "-")
    base = "".join(ch for ch in base if ch.isalnum() or ch in ("-","_","."))
    ts = int(time.time()*1000)
    name = f"{base}-{ts}.png"
    path = os.path.join(img_dir, name)
    # load with PIL and save as PNG on 800x600 canvas
    try:
        img = Image.open(file).convert("RGB")
    except Exception:
        # if bytes, read from buffer
        img = Image.open(io.BytesIO(file.getvalue())).convert("RGB")
    img.thumbnail((900,900))
    canvas = Image.new("RGB",(800,600),(245,245,245))
    iw, ih = img.size
    canvas.paste(img, ((800-iw)//2,(600-ih)//2))
    canvas.save(path, "PNG", optimize=True)
    return f"images/{name}"

# ---------------- Pages ----------------
def page_market():
    st.header("🛒 Marketplace")
    df=load_products(); vec,X = build_vectors(df)
    with st.expander("🧪 Skin-Type Quick Set"):
        opt = st.selectbox("Choose", ["all","normal","dry","oily","sensitive","acne-prone"])
        if st.button("Use this skin type"): st.session_state.skin_type=opt; st.success(f"Skin type set: {opt}")
    c1,c2 = st.columns([3,2])
    with c1: q = st.text_input("Search (e.g., 'acne tea tree gel')","")
    with c2: cat = st.multiselect("Category", sorted(df["category"].unique().tolist()))
    recs = recommend(df, vec, X, q, st.session_state.get("skin_type","all"))
    if cat: recs = recs[recs["category"].isin(cat)]
    for _, row in recs.iterrows():
        cc1, cc2 = st.columns([1,2], vertical_alignment="center")
        with cc1:
            img_path = os.path.join(BASE_DIR, row["image_path"])
            try:
                img = Image.open(img_path); st.image(img, use_column_width=True)
            except: st.warning(f"🖼️ Image missing: {img_path}")
        with cc2:
            st.subheader(f"{row['name']} · ₹{row['price']:.0f}")
            st.write(f"⭐ {row['rating']} | {row['category']} | Stock: {int(row['stock'])} | Seller: {row['seller']}")
            st.write(row["description"])
            st.caption("Ingredients: "+str(row["ingredients"]))
            st.info(ingredient_advice(row["ingredients"], st.session_state.get("skin_type","all")))
            b1,b2,b3 = st.columns([1,1,1])
            with b1:
                qty = st.number_input(f"Qty #{row['id']}", 1, 10, 1, key=f'qty_{row["id"]}_{row["name"]}')
            with b2:
                if st.button("Add to Cart", key=f'add_{row["id"]}_{row["name"]}'):
                    cart = st.session_state.setdefault("cart", [])
                    cart.append({"id":int(row["id"]), "name":row["name"], "price":float(row["price"]), "qty":int(qty)})
                    st.success("Added to cart.")
            with b3:
                if st.button("Chat with Seller", key=f'chat_{row["id"]}_{row["name"]}'):
                    st.session_state["chat_product_id"]=int(row["id"]); st.session_state["chat_seller"]=row["seller"]
                    st.session_state["page"]="chat"; st.rerun()

def page_cart():
    st.header("🧺 Your Cart")
    cart = st.session_state.get("cart", [])
    if not cart: st.info("Your cart is empty."); return
    df=pd.DataFrame(cart); df["line_total"]=df["qty"]*df["price"]
    st.table(df[["name","qty","price","line_total"]])
    total=float(df["line_total"].sum()); st.success(f"Subtotal: ₹{total:.0f}")
    st.subheader("Checkout")
    name = st.text_input("Full Name", value=st.session_state.get("buyer_name",""))
    addr = st.text_area("Shipping Address", value=st.session_state.get("buyer_addr",""))
    pay = st.selectbox("Payment Method", ["Cash on Delivery (COD)","UPI","Card"])
    if st.button("Place Order ✅"):
        if not name or not addr: st.error("Please fill name & address.")
        else:
            st.session_state["buyer_name"]=name; st.session_state["buyer_addr"]=addr
            user = st.session_state.get("user", {"email":"buyer@demo"})
            for it in cart: place_order(user["email"], it["id"], it["qty"], it["price"], payment_mode=pay)
            st.session_state["cart"]=[]; st.success(f"🎉 Order placed! Ref: SM-{int(time.time())}"); st.balloons()

def page_chat():
    st.header("💬 Chat with Seller")
    user = st.session_state.get("user", {"email":"buyer@demo"})
    seller = st.session_state.get("chat_seller","HerbaCare")
    pid = st.session_state.get("chat_product_id",None)
    if pid:
        p = load_products().set_index("id").loc[int(pid)]
        st.write(f"**About:** {p['name']} (₹{p['price']:.0f}) · Seller: {p['seller']}")
    hist = get_chat(user["email"], seller, pid)
    for _,r in hist.iterrows():
        who = "🧑 Buyer" if r["sender"]=="buyer" else ("🤖 Bot" if r["sender"]=="bot" else "🧑‍💼 Seller")
        st.write(f"**{who}:** {r['message']}  \n_{r['ts']}_")
    msg = st.text_input("Type your message")
    if st.button("Send"):
        if msg.strip():
            save_chat(user["email"], seller, pid, "buyer", msg.strip())
            row = None
            if pid:
                d = load_products().set_index("id")
                if int(pid) in d.index: row = d.loc[int(pid)].to_dict()
            ans = reply(intent(msg), row, st.session_state.get("skin_type","all"))
            save_chat(user["email"], seller, pid, "bot", ans); st.rerun()

def seller_dashboard():
    st.header("🧑‍💼 Seller Dashboard")
    user = st.session_state.get("user", {"email":"seller@demo","role":"seller"})
    seller_name = user.get("email","seller@demo").split("@")[0].title()
    st.caption(f"Logged in as: **{user.get('email','seller@demo')}**")

    tabs = st.tabs(["📦 My Products", "➕ Add Product (with Image Upload)", "📬 My Orders", "💬 Chats"])

    # My Products
    with tabs[0]:
        df = load_products()
        mine = df[df["seller"].str.lower()==seller_name.lower()]
        st.subheader("Your Products")
        if mine.empty: st.info("No products yet.")
        else: st.dataframe(mine, use_container_width=True)

    # Add Product with Image Upload
    with tabs[1]:
        st.subheader("Add Product")
        name = st.text_input("Product name")
        category = st.selectbox("Category", ["Cleansers","Moisturizers","Masks","Toners","Serums","Sunscreen","Other"])
        price = st.number_input("Price (₹)", 1, 10000, 299)
        stock = st.number_input("Stock", 0, 100000, 50)
        rating = st.slider("Rating", 0.0, 5.0, 4.3, 0.1)
        ingredients = st.text_input("Ingredients (comma separated)")
        desc = st.text_area("Short description")
        img_file = st.file_uploader("Upload product image", type=["png","jpg","jpeg"])
        if st.button("Create Product ✅"):
            if not name or not img_file:
                st.error("Please enter a name and upload an image.")
            else:
                # Save image into /images and store relative path
                safe_name = name if isinstance(name,str) else "product"
                rel_path = save_uploaded_image(img_file, safe_name)
                pid = next_product_id()
                with engine.begin() as c:
                    c.exec_driver_sql("""
                    INSERT INTO products(id,name,category,price,stock,rating,ingredients,description,image_path,seller,skin_types)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (pid, name, category, float(price), int(stock), float(rating), ingredients, desc, rel_path, seller_name, "all"))
                st.success("Product added."); st.rerun()

    # Orders
    with tabs[2]:
        st.subheader("Orders")
        with engine.begin() as c:
            prods = pd.read_sql("SELECT * FROM products", c)
            ords = pd.read_sql("SELECT * FROM orders", c)
        if prods.empty or ords.empty: st.info("No orders yet.")
        else:
            merged = ords.merge(prods[["id","name","seller","price"]], left_on="product_id", right_on="id", how="left")
            mine = merged[ merged["seller"].str.lower()==seller_name.lower() ]
            if mine.empty: st.info("No orders for your products yet.")
            else:
                st.dataframe(mine[["id_x","buyer_email","name","qty","total","payment_mode","ts"]].rename(columns={"id_x":"order_id"}), use_container_width=True)

    # Chats
    with tabs[3]:
        st.subheader("Customer Chats")
        with engine.begin() as c:
            chats = pd.read_sql("SELECT * FROM chats ORDER BY id DESC LIMIT 200", c)
        subset = chats[chats["seller"].str.lower()==seller_name.lower()].sort_values("id", ascending=True)
        if subset.empty:
            st.info("No chats yet.")
        else:
            for _,r in subset.iterrows():
                who = "🧑 Buyer" if r["sender"]=="buyer" else ("🤖 Bot" if r["sender"]=="bot" else "🧑‍💼 Seller")
                st.write(f"**{who}:** {r['message']}  \n_{r['ts']}_")

def page_admin():
    st.header("🛠️ Admin")
    with engine.begin() as c:
        prods = pd.read_sql("SELECT * FROM products", c)
        ords = pd.read_sql("SELECT * FROM orders", c)
        users = pd.read_sql("SELECT id,email,role FROM users", c)
    st.subheader("Products"); st.dataframe(prods, use_container_width=True)
    st.subheader("Orders"); st.dataframe(ords, use_container_width=True)
    st.subheader("Users"); st.dataframe(users, use_container_width=True)

# ---------------- App Shell ----------------
def navbar():
    st.markdown("<style>.stButton button{border-radius:8px;padding:8px 16px;font-weight:600}</style>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns([3,1,1,1])
    with c1: st.title("🪴 AI Herbal Skincare")
    with c2:
        if st.button("🏠 Marketplace"): st.session_state["page"]="market"; st.rerun()
    with c3:
        if st.button("🧺 View Cart"): st.session_state["page"]="cart"; st.rerun()
    with c4:
        if st.button("💬 Chat"): st.session_state["page"]="chat"; st.rerun()

def run_app():
    init_db()
    auth_box()
    navbar()

    page = st.session_state.get("page","market")
    tabs = st.tabs(["Buyer","Seller","Admin"])

    with tabs[0]:
        if page=="market": page_market()
        elif page=="cart": page_cart()
        elif page=="chat": page_chat()

    with tabs[1]:
        role = st.session_state.get("user",{}).get("role","buyer")
        if role in ("seller","admin"): seller_dashboard()
        else: st.info("Sign in as seller to access: seller@demo / seller")

    with tabs[2]:
        if st.session_state.get("user",{}).get("role")=="admin": page_admin()
        else: st.info("Sign in as admin to access: admin@demo / admin")

if __name__ == "__main__":
    run_app()
