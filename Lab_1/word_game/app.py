from flask import Flask, render_template, request, redirect, url_for, session
from nltk.corpus import wordnet as wn
import nltk, random, os


nltk.download('wordnet')
nltk.download('omw-1.4') 

app = Flask(__name__)
# app.secret_key = "claudiu_chichirau"
app.secret_key = os.urandom(24)


WORDS = [
    "car", "tree", "dog", "computer", "house", "music", "book", "phone", "river", "mountain",
    "city", "bird", "food", "teacher", "school", "movie", "flower", "chair", "road", "rain",
    "cat", "coffee", "planet", "train", "bicycle", "child", "friend", "art", "doctor", "dream",
    "game", "garden", "ocean", "star", "sun", "moon", "forest", "island", "light", "shadow",
    "time", "wind", "fire", "snow", "ice", "heart", "smile", "laughter", "music", "dance",
    "love", "peace", "hope", "faith", "courage", "strength", "wisdom", "freedom", "justice",
    "honor", "truth", "beauty", "nature", "adventure", "journey", "memory", "history", "future", "dream"
]

def compute_similarity(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    if not synsets1 or not synsets2:
        return 0.0

    max_sim = 0
    for s1 in synsets1:
        for s2 in synsets2:
            sim = s1.wup_similarity(s2) 
            if sim and sim > max_sim:
                max_sim = sim
    return max_sim




@app.route("/", methods=["GET", "POST"])
def index():
    session["words"] = random.sample(WORDS, 5)
    session["current"] = 0
    session["score_total"] = 0
    return redirect(url_for("game"))



@app.route("/game", methods=["GET", "POST"])
def game():
    if "words" not in session:
        return redirect(url_for("index"))

    words = session["words"]
    current = session.get("current", 0)
    score_total = session.get("score_total", 0)

    if current >= 5:
        return redirect(url_for("result"))

    base_word = words[current]

    feedback = session.pop("feedback", "")
    round_score = session.pop("round_score", None)

    if request.method == "POST":
        user_word = request.form["user_word"].strip().lower()
        similarity = compute_similarity(base_word, user_word)

        round_score = int(similarity * 100)
        score_total += round_score
        session["score_total"] = score_total

        if similarity >= 0.8:
            feedback = f"Excelent! '{user_word}' este foarte apropiat de '{base_word}' ({similarity:.2f})."
        elif similarity > 0.5:
            feedback = f"Bine! '{user_word}' este destul de legat de '{base_word}' ({similarity:.2f})."
        elif similarity > 0.2:
            feedback = f"Slabut... '{user_word}' are o legatura mica cu '{base_word}' ({similarity:.2f})."
        else:
            feedback = f"Nicio legatura semantica intre '{user_word}' si '{base_word}' ({similarity:.2f})."

        session["feedback"] = feedback
        session["round_score"] = round_score

        session["current"] = current + 1

        return redirect(url_for("game"))


    return render_template(
        "index.html",
        base_word=base_word,
        current=current + 1,
        total=5,
        feedback=feedback,
        score_total=score_total,
        round_score=round_score,
    )


@app.route("/result")
def result():
    score_total = session.get("score_total", 0)
    if score_total >= 400:
        message = "🏆 Excelent! Esti un adevarat maestru al cuvintelor!"
        color_class = "green"
    elif score_total >= 250:
        message = "👏 Bine jucat! Ai o buna intuitie semantica!"
        color_class = "blue"
    else:
        message = "🙂 Poti mai bine! Incearca din nou!"
        color_class = "red"

    return render_template("result.html", score_total=score_total, message=message, color_class=color_class)



if __name__ == "__main__":
    app.run(debug=True)