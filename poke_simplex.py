import numpy as np
import itertools

class PokeSimplex:
    def __init__(self, c, A, b, operators=None, maximize=True):
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.operators = operators if operators else ["<="] * len(b)
        self.maximize = maximize
        self.num_vars = len(c)
        self.num_constraints = len(b)
        
        self.iterations_log = []
        self.M = 1e9 
        self.col_headers = [] 
        
        self.tableau = None
        self.basic_vars = [] 
        
    def initialize_tableau(self):
        # 1. Definir Estructura de Columnas
        self.col_headers = [f"x{i+1}" for i in range(self.num_vars)]
        extra_cols_map = [] 
        
        s_cnt, e_cnt, a_cnt = 1, 1, 1
        
        for op in self.operators:
            if op == "<=":
                extra_cols_map.append({'type': 's', 'name': f"s{s_cnt}"})
                self.col_headers.append(f"s{s_cnt}")
                s_cnt += 1
            elif op == ">=":
                extra_cols_map.append({'type': 'e_a', 'name_e': f"e{e_cnt}", 'name_a': f"a{a_cnt}"})
                self.col_headers.append(f"e{e_cnt}")
                self.col_headers.append(f"a{a_cnt}")
                e_cnt += 1
                a_cnt += 1
            elif op == "=":
                extra_cols_map.append({'type': 'a', 'name': f"a{a_cnt}"})
                self.col_headers.append(f"a{a_cnt}")
                a_cnt += 1
                
        self.col_headers.append("Sol") 
        
        # 2. Inicializar Matriz
        n_cols = len(self.col_headers)
        n_rows = self.num_constraints + 1
        self.tableau = np.zeros((n_rows, n_cols))
        
        self.tableau[:-1, :self.num_vars] = self.A
        self.tableau[:-1, -1] = self.b
        
        # 3. Llenar variables básicas
        self.basic_vars = []
        for i, struct in enumerate(extra_cols_map):
            if struct['type'] == 's':
                col_idx = self.col_headers.index(struct['name'])
                self.tableau[i, col_idx] = 1
                self.basic_vars.append(col_idx)
            elif struct['type'] == 'e_a':
                col_e = self.col_headers.index(struct['name_e'])
                col_a = self.col_headers.index(struct['name_a'])
                self.tableau[i, col_e] = -1
                self.tableau[i, col_a] = 1
                self.basic_vars.append(col_a) 
            elif struct['type'] == 'a':
                col_idx = self.col_headers.index(struct['name'])
                self.tableau[i, col_idx] = 1
                self.basic_vars.append(col_idx)

        # 4. Fila Z Inicial
        if self.maximize:
            self.tableau[-1, :self.num_vars] = -self.c
        else:
            self.tableau[-1, :self.num_vars] = self.c

        # 5. Penalización M Grande
        for i, var_idx in enumerate(self.basic_vars):
            col_name = self.col_headers[var_idx]
            if col_name.startswith('a'):
                if self.maximize:
                    self.tableau[-1, :] -= self.M * self.tableau[i, :]
                else:
                    self.tableau[-1, :] += self.M * self.tableau[i, :]

    # --- MODIFICACIÓN CLAVE AQUÍ ---
    def solve(self, skip_initialization=False):
        if not skip_initialization:
            self.initialize_tableau()
            
        max_iter = 100
        
        for i in range(max_iter):
            snapshot = self._get_snapshot(i)
            z_row = self.tableau[-1, :-1]
            
            # Optimalidad
            is_optimal = False
            pivot_col = -1
            
            if self.maximize:
                if np.all(z_row >= -1e-5): is_optimal = True
                else: pivot_col = np.argmin(z_row)
            else:
                if np.all(z_row <= 1e-5): is_optimal = True
                else: pivot_col = np.argmax(z_row)

            if is_optimal:
                snapshot['status'] = "OPTIMAL"
                if self._check_infeasibility():
                    snapshot['status'] = "INFEASIBLE"
                    snapshot['message'] = "¡Región Factible Vacía! (Infactible)"
                else:
                    snapshot['message'] = "¡Solución Óptima encontrada!"
                self.iterations_log.append(snapshot)
                break
                
            snapshot['status'] = "IN_PROGRESS"
            
            # Ratio Test
            rhs = self.tableau[:-1, -1]
            col_vals = self.tableau[:-1, pivot_col]
            
            ratios = []
            for idx, val in enumerate(col_vals):
                if val > 1e-9: 
                    ratios.append(rhs[idx] / val)
                else:
                    ratios.append(np.inf)
            
            if all(r == np.inf for r in ratios):
                snapshot['status'] = "UNBOUNDED"
                snapshot['message'] = "Solución No Acotada (Infinita)."
                self.iterations_log.append(snapshot)
                return self.iterations_log
                
            pivot_row = np.argmin(ratios)
            
            snapshot['entering_var'] = self.col_headers[pivot_col]
            snapshot['leaving_var'] = self.col_headers[self.basic_vars[pivot_row]]
            snapshot['pivot_row'] = int(pivot_row)
            snapshot['pivot_col'] = int(pivot_col)
            
            self.iterations_log.append(snapshot)
            
            self._pivot(pivot_row, pivot_col)
            self.basic_vars[pivot_row] = pivot_col
            
        return self.iterations_log

    def _pivot(self, row, col):
        pivot_val = self.tableau[row, col]
        self.tableau[row, :] /= pivot_val
        for r in range(self.tableau.shape[0]):
            if r != row:
                factor = self.tableau[r, col]
                self.tableau[r, :] -= factor * self.tableau[row, :]
                
    def _check_infeasibility(self):
        # Verifica si hay artificiales en la base con valor > 0
        for i, var_idx in enumerate(self.basic_vars):
            if var_idx < len(self.col_headers):
                name = self.col_headers[var_idx]
                val = self.tableau[i, -1]
                if name.startswith('a') and val > 1e-5:
                    return True
        return False

    def _get_snapshot(self, iter_num):
        basic_names = [self.col_headers[i] for i in self.basic_vars]
        z_val = self.tableau[-1, -1]
        
        # En Simplex tabla completa, a veces Z se muestra directo
        return {
            "iteration": iter_num,
            "z_value": f"{z_val:.2e}" if abs(z_val) > 1e6 else round(z_val, 4),
            "basic_vars": basic_names,
            "headers": self.col_headers,
            "tableau": self.tableau.tolist()
        }

class PokeTwoPhase:
    def __init__(self, c, A, b, operators=None, maximize=True):
        self.c_original = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.operators = operators if operators else ["<="] * len(b)
        self.maximize_original = maximize
        self.log = []
        
    def solve(self):
        # --- FASE 1 ---
        num_art = sum(1 for op in self.operators if op in [">=", "="])
        
        if num_art == 0:
            solver = PokeSimplex(self.c_original, self.A, self.b, self.operators, self.maximize_original)
            return solver.solve()

        # Configurar Fase 1
        # CORRECCIÓN: Usamos maximize=True. 
        # Matemáticamente, Minimizar W es equivalente a Maximizar -W.
        # Al restar las filas artificiales de Z, obtenemos la fila correcta para Maximización.
        c_phase1 = np.zeros(len(self.c_original))
        solver_f1 = PokeSimplex(c_phase1, self.A, self.b, self.operators, maximize=True) # <--- CAMBIO AQUÍ (True)
        solver_f1.initialize_tableau()
        
        # Ajustar Fila W (Z de Fase 1)
        solver_f1.tableau[-1, :] = 0
        art_indices = [i for i, name in enumerate(solver_f1.col_headers) if name.startswith('a')]
        
        # Ponemos 1s en las artificiales (Representa Z + Sum(a_i) = 0)
        for idx in art_indices:
            solver_f1.tableau[-1, idx] = 1 
            
        # Price out (Hacer ceros en la base)
        for row_idx in range(solver_f1.num_constraints):
            basic_idx = solver_f1.basic_vars[row_idx]
            if basic_idx in art_indices:
                solver_f1.tableau[-1, :] -= solver_f1.tableau[row_idx, :]
                
        steps_f1 = solver_f1.solve(skip_initialization=True)
        
        for step in steps_f1:
            step['phase'] = "FASE 1"
            # Ajuste visual: Como estamos maximizando -W, el valor Z saldrá negativo (ej: -9).
            # Para que el usuario no se confunda, podemos mostrarlo positivo en el log si gustas,
            # pero matemáticamente es correcto que sea negativo en la tabla.
            self.log.append(step)
            
        # --- TRANSICIÓN ---
        last_step = steps_f1[-1]
        w_val = last_step['tableau'][-1][-1]
        
        # Validamos si W es 0 (o -0). Usamos abs() por seguridad.
        if abs(w_val) > 1e-4:
            self.log.append({
                "status": "INFEASIBLE", "phase": "TRANSICIÓN",
                "message": f" INFACTIBLE: La Fase 1 terminó con penalización W = {abs(w_val):.2f} > 0."
            })
            return self.log
            
        self.log.append({
            "status": "PHASE_TRANSITION", "phase": "TRANSICIÓN",
            "message": " FASE 1 COMPLETADA (W=0). Iniciando Fase 2...",
            "tableau": []
        })
        
        # --- FASE 2 ---
        current_tableau = np.array(last_step['tableau'])
        current_headers = last_step['headers']
        current_basic = solver_f1.basic_vars
        
        keep_indices = [i for i, name in enumerate(current_headers) if not name.startswith('a')]
        new_tableau = current_tableau[:, keep_indices]
        new_headers = [current_headers[i] for i in keep_indices]
        
        new_basic_vars = []
        for old_idx in current_basic:
            name = current_headers[old_idx]
            if name in new_headers:
                new_idx = new_headers.index(name)
                new_basic_vars.append(new_idx)
            else:
                found_replacement = False
                for j, col_name in enumerate(new_headers):
                    if j not in new_basic_vars:
                        new_basic_vars.append(j)
                        found_replacement = True
                        break

        # Restaurar Z Original
        new_tableau[-1, :] = 0
        if self.maximize_original:
            new_tableau[-1, :len(self.c_original)] = -self.c_original
        else:
            new_tableau[-1, :len(self.c_original)] = self.c_original
            
        for r, col_basic in enumerate(new_basic_vars):
            z_coef = new_tableau[-1, col_basic]
            if abs(z_coef) > 1e-9:
                new_tableau[-1, :] -= z_coef * new_tableau[r, :]
                
        solver_f2 = PokeSimplex(self.c_original, [], [], self.operators, self.maximize_original)
        solver_f2.tableau = new_tableau
        solver_f2.col_headers = new_headers
        solver_f2.basic_vars = new_basic_vars
        
        steps_f2 = solver_f2.solve(skip_initialization=True)
        
        for step in steps_f2:
            step['phase'] = "FASE 2"
            step['iteration'] += len(steps_f1)
            self.log.append(step)
            
        return self.log
    
class PokeAlgebraic:
    def __init__(self, c, A, b, operators=None, maximize=True):
        # Reutilizamos PokeSimplex para convertir el problema a forma estándar (Igualdades)
        # Esto nos ahorra reescribir la lógica de agregar s1, e1, a1, etc.
        self.standardizer = PokeSimplex(c, A, b, operators, maximize)
        self.maximize = maximize
        
    def solve(self):
        # 1. Obtener el sistema en forma estándar (Matriz completa)
        self.standardizer.initialize_tableau()
        tableau = self.standardizer.tableau
        headers = self.standardizer.col_headers
        
        # Separar la matriz A extendida y el vector b de la tabla simplex
        # La tabla tiene la forma: [ A_ext | b ]
        #                          [  Z    | z ]
        
        num_rows = self.standardizer.num_constraints
        num_cols = len(headers) - 1 # Ignorando columna Sol
        
        # Matriz A de coeficientes (incluyendo holguras/artificiales)
        matrix_A = tableau[:num_rows, :num_cols]
        # Vector b
        vector_b = tableau[:num_rows, -1]
        
        # Función Objetivo (Coeficientes originales en la tabla Z están negativos para Max)
        # Los recuperamos de la fila Z
        z_row = tableau[-1, :num_cols]
        
        solutions_log = []
        
        # 2. Combinatoria: Seleccionar 'm' columnas para ser Bases
        # n_total = columnas totales (variables decisión + holguras + artificiales)
        # m = número de restricciones
        
        indices = list(range(num_cols))
        combinations = list(itertools.combinations(indices, num_rows))
        
        iteration_count = 1
        
        for basic_indices in combinations:
            # Construir submatriz B
            try:
                B = matrix_A[:, basic_indices]
                
                # Calcular determinante para ver si es singular
                det = np.linalg.det(B)
                
                if abs(det) < 1e-9:
                    # Sistema singular (no tiene solución única)
                    continue 
                    
                # Resolver sistema B * X_b = b
                X_b = np.linalg.solve(B, vector_b)
                
                # Construir solución completa (llenar ceros en no básicas)
                full_solution = np.zeros(num_cols)
                for i, idx in enumerate(basic_indices):
                    full_solution[idx] = X_b[i]
                    
                # 3. Clasificar Solución [Req 3.2.4]
                # Factible si todas las variables >= 0
                is_feasible = np.all(full_solution >= -1e-9)
                
                # Calcular Z
                # Z = c * x. (Usamos la fila Z del tableau. Nota: en tableau Max es -c, en Min es c)
                # Z_val = - (z_row dot solution) si es Max porque z_row tiene -c
                # Z_val = (z_row dot solution) si es Min porque z_row tiene c
                # Ajustamos la penalización M si aplica
                
                current_z = 0
                if self.maximize:
                     current_z = -np.dot(z_row, full_solution)
                else:
                     current_z = np.dot(z_row, full_solution)
                     
                # Filtrar valores Z gigantes (infactibles por M grande)
                if abs(current_z) > 1e8:
                    status = "Infactible (M)"
                    is_feasible = False
                else:
                    status = "Factible" if is_feasible else "No Factible"

                # Formatear para salida
                point_vars = {}
                for idx, val in enumerate(full_solution):
                    # Solo mostramos variables con valor > 0 o relevantes
                    if abs(val) > 1e-5:
                        point_vars[headers[idx]] = round(val, 4)
                
                solutions_log.append({
                    "id": iteration_count,
                    "basic_indices": [headers[i] for i in basic_indices],
                    "values": point_vars,
                    "z_value": round(current_z, 4),
                    "status": status,
                    "is_optimal": False # Se determinará al final
                })
                
                iteration_count += 1
                
            except np.linalg.LinAlgError:
                continue # Matriz no invertible
                
        # 4. Identificar el Óptimo
        if not solutions_log:
            return []

        feasible_solutions = [s for s in solutions_log if s['status'] == "Factible"]
        
        if feasible_solutions:
            if self.maximize:
                best_sol = max(feasible_solutions, key=lambda x: x['z_value'])
            else:
                best_sol = min(feasible_solutions, key=lambda x: x['z_value'])
                
            # Marcar el óptimo
            for sol in solutions_log:
                if sol == best_sol:
                    sol['is_optimal'] = True
                    sol['status'] = "OPTIMO"

        return solutions_log
    
class PokeRevisedSimplex:
    def __init__(self, c, A, b, operators=None, maximize=True):
        # Usamos el constructor estándar para preparar la matriz A completa (con slacks/artificiales)
        self.helper = PokeSimplex(c, A, b, operators, maximize)
        self.maximize = maximize
        self.log = []
        
    def solve(self):
        # 1. Preparación de Datos Matriciales
        self.helper.initialize_tableau()
        
        # Extraer componentes de la tabla inicial generada
        full_headers = self.helper.col_headers
        num_rows = self.helper.num_constraints
        num_cols_total = len(full_headers) - 1 # Sin RHS
        
        # Matriz A completa (incluyendo holguras/artificiales)
        # OJO: PokeSimplex pone la matriz en las primeras filas
        matrix_A_full = self.helper.tableau[:num_rows, :num_cols_total]
        
        # Vector b (Lado derecho)
        vector_b = self.helper.tableau[:num_rows, -1]
        
        # Vector c (Costos)
        # Nota: En PokeSimplex la fila Z tiene -c (para Max) o c (para Min).
        # Recuperamos los costos originales C_j para la fórmula (Zj - Cj) = Cb * B_inv * Aj - Cj
        # Fila Z inicial = [ -c_originales ... 0 ... ] + Penalizaciones M
        # Es más seguro reconstruir el vector c basado en los headers y la M.
        
        vector_c = np.zeros(num_cols_total)
        
        # Reconstruir costos incluyendo M
        M_val = 1e9
        for j, name in enumerate(full_headers[:-1]): # Excluir 'Sol'
            # Costos originales (Variables de decisión)
            if j < self.helper.num_vars:
                vector_c[j] = self.helper.c[j]
            
            # Penalización M para artificiales
            if name.startswith('a'):
                # Si Max: penalizamos con -M. Si Min: +M.
                vector_c[j] = -M_val if self.maximize else M_val
            
            # Slacks (s) y Excess (e) tienen costo 0, ya inicializado.

        # 2. Base Inicial
        # PokeSimplex ya calculó una base inicial factible (con holguras/artificiales)
        basic_indices = list(self.helper.basic_vars) # Copia de índices
        
        max_iter = 100
        for it in range(max_iter):
            # Paso A: Matriz Base B
            # Extraemos las columnas correspondientes a la base actual
            B = matrix_A_full[:, basic_indices]
            
            try:
                # Paso B: Inversa de B [Cite: 47 - Mostrar B^-1]
                B_inv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                self.log.append({"status": "ERROR", "message": "Matriz Base Singular (No invertible) [Cite: 66]"})
                return self.log

            # Paso C: Vector de Multiplicadores Simplex (pi) [Cite: 47]
            # pi = c_B * B_inv
            c_B = vector_c[basic_indices]
            pi = np.dot(c_B, B_inv)
            
            # Paso D: Calcular Costos Reducidos (Zj - Cj) para NO básicas
            # reduced_cost_j = pi * A_j - c_j
            # Para Max: Buscamos el más negativo (o positivo según convención Zj-Cj vs Cj-Zj)
            # Convención estándar Max: Entra la variable con Cj - Zj > 0 más grande.
            # O equivalentemente: Zj - Cj < 0 más negativo.
            # Zj = pi * Aj. -> Entra si pi*Aj - Cj < 0.
            
            entering_idx = -1
            best_rc = 0
            
            reduced_costs_log = {} # Para mostrar al usuario
            
            optimal = True
            
            for j in range(num_cols_total):
                if j not in basic_indices:
                    A_j = matrix_A_full[:, j]
                    z_j = np.dot(pi, A_j)
                    rc = z_j - vector_c[j] # Zj - Cj
                    
                    reduced_costs_log[full_headers[j]] = rc
                    
                    if self.maximize:
                        # Maximizar: Mejorar si Zj - Cj < 0 (Cj > Zj cost marginal positivo)
                        # Usando tolerancia numérica
                        if rc < -1e-5:
                            optimal = False
                            if rc < best_rc:
                                best_rc = rc
                                entering_idx = j
                    else:
                        # Minimizar: Mejorar si Zj - Cj > 0
                        if rc > 1e-5:
                            optimal = False
                            if rc > best_rc:
                                best_rc = rc
                                entering_idx = j

            # Calcular Solución Actual (X_b = B_inv * b) [Cite: 47]
            x_b = np.dot(B_inv, vector_b)
            
            # Calcular valor Z actual
            z_val = np.dot(c_B, x_b)

            # --- SNAPSHOT (Reporte Iteración) ---
            snapshot = {
                "iteration": it,
                "z_value": f"{z_val:.2e}" if abs(z_val) > 1e6 else round(z_val, 4),
                "basic_vars": [full_headers[i] for i in basic_indices],
                "pi_vector": pi.tolist(), # [Cite: 47]
                "b_inv": B_inv.tolist(),  # [Cite: 47]
                "current_sol": x_b.tolist(),
                "reduced_costs": {k: round(v, 4) for k, v in reduced_costs_log.items()}
            }

            if optimal:
                snapshot['status'] = "OPTIMAL"
                snapshot['message'] = "¡Análisis completo! No hay costos reducidos favorables."
                self.log.append(snapshot)
                break
            
            snapshot['status'] = "IN_PROGRESS"
            snapshot['entering_var'] = full_headers[entering_idx]
            
            # Paso E: Columna Pivote Actualizada (alpha = B_inv * A_k)
            # A_k es la columna original de la variable que entra
            A_k = matrix_A_full[:, entering_idx]
            alpha = np.dot(B_inv, A_k)
            
            # Paso F: Prueba del Cociente (Ratio Test)
            # Min(X_b_i / alpha_i) para alpha_i > 0
            ratios = []
            valid_ratios_indices = []
            
            for i, val in enumerate(alpha):
                if val > 1e-9:
                    r = x_b[i] / val
                    ratios.append(r)
                    valid_ratios_indices.append(i)
                else:
                    ratios.append(np.inf)
            
            if not valid_ratios_indices:
                snapshot['status'] = "UNBOUNDED"
                snapshot['message'] = "Solución No Acotada detectada por Mewtwo."
                self.log.append(snapshot)
                return self.log
            
            # Encontrar índice (en la base local) que sale
            min_ratio = min([ratios[i] for i in valid_ratios_indices])
            # Obtener el índice local (0..m-1) que corresponde al min_ratio
            # Necesitamos tener cuidado con empates (Degeneración) [Cite: 63]
            pivot_row_local = -1
            for i in valid_ratios_indices:
                if ratios[i] == min_ratio:
                    pivot_row_local = i
                    break
            
            leaving_var_real_idx = basic_indices[pivot_row_local]
            snapshot['leaving_var'] = full_headers[leaving_var_real_idx]
            
            self.log.append(snapshot)
            
            # Paso G: Actualizar Base
            basic_indices[pivot_row_local] = entering_idx
            
        return self.log