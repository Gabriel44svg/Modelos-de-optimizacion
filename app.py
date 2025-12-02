from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from poke_simplex import PokeSimplex, PokeTwoPhase, PokeAlgebraic, PokeRevisedSimplex  # <--- Importar ambos motores

# Configuración para servir archivos desde la carpeta raíz
app = Flask(__name__, template_folder='.')

# Habilitar CORS para permitir peticiones desde Live Server (VSCode)
CORS(app)


# ============================
#      RUTAS DE VISTAS
# ============================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/simplex.html')
def simplex_view():
    return render_template('simplex.html')

@app.route('/mgrande.html')
def mgrande_view():
    return render_template('mgrande.html')

@app.route('/dosfases.html')     # <--- NUEVA RUTA
def dosfases_view():
    return render_template('dosfases.html')

@app.route('/algebraico.html')
def algebraico_view():
    return render_template('algebraico.html')

@app.route('/simplexrevisado.html')
def revisado_view():
    return render_template('simplexrevisado.html')

# ============================
#       API DE CÁLCULO
# ============================
@app.route('/api/solve_simplex', methods=['POST'])
def solve_api():
    print("--- Nueva Petición Recibida ---")
    data = request.json

    # Objetivo: max o min
    obj_type = data.get('objective', 'max')
    is_maximize = (obj_type == 'max')

    # Operadores: <=, >=, =
    operators = data.get('operators')

    # Método seleccionado: simplex | mgrande | dosfases | algebraico
    method = data.get('method', 'simplex')

    print(f"Objetivo: {obj_type}")
    print(f"Operadores: {operators}")
    print(f"Método elegido: {method}")

    try:
        iteraciones = []

        # ---------------------------------------------------
        #               MÉTODO ALGEBRAICO
        # ---------------------------------------------------
        if method == 'algebraico':
            solver = PokeAlgebraic(
                c=data['c'],
                A=data['A'],
                b=data['b'],
                operators=operators,
                maximize=is_maximize
            )
            iteraciones = solver.solve()

            # Enviar flag especial para el frontend
            return jsonify({
                "success": True,
                "type": "algebraic",
                "steps": iteraciones
            })

         # ---------------------------------------------------
        #               MÉTODO REVISADO (NUEVO)
        # ---------------------------------------------------
        if method == 'revisado':
            solver = PokeRevisedSimplex(
                c=data['c'],
                A=data['A'],
                b=data['b'],
                operators=operators,
                maximize=is_maximize
            )
            iteraciones = solver.solve()

            # Flag para render especial
            return jsonify({
                "success": True,
                "type": "revised",
                "steps": iteraciones
            })

        # ---------------------------------------------------
        #               MÉTODO DE DOS FASES
        # ---------------------------------------------------
        if method == 'dosfases':
            solver = PokeTwoPhase(
                c=data['c'],
                A=data['A'],
                b=data['b'],
                operators=operators,
                maximize=is_maximize
            )
            iteraciones = solver.solve()

        else:
            # ---------------------------------------------------
            #     SIMPLEX ESTÁNDAR O M-GRANDE (automático)
            # ---------------------------------------------------
            solver = PokeSimplex(
                c=data['c'],
                A=data['A'],
                b=data['b'],
                operators=operators,
                maximize=is_maximize
            )
            iteraciones = solver.solve()

        return jsonify({"success": True, "steps": iteraciones})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)})


# ============================
#       EJECUCIÓN
# ============================
if __name__ == '__main__':
    print("Servidor Poke-Optimizador LISTO en http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
