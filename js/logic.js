// Lógica del Poké-Optimizador - Validación Estricta

function generarMatriz() {
    // ... (El código de generarMatriz se queda IGUAL que antes) ...
    const numVars = parseInt(document.getElementById('num-vars').value);
    const numCons = parseInt(document.getElementById('num-cons').value);
    const modelArea = document.getElementById('model-area');
    
    const zContainer = document.getElementById('z-function');
    const matrixContainer = document.getElementById('matrix-container');
    zContainer.innerHTML = 'Z = ';
    matrixContainer.innerHTML = '';

    // Generar Inputs Z
    for (let i = 1; i <= numVars; i++) {
        zContainer.innerHTML += `
            <input type="number" id="z_x${i}" placeholder="c${i}"> 
            <label>x<sub>${i}</sub></label>
            ${i < numVars ? ' + ' : ''}
        `;
    }

    // Generar Restricciones
    for (let i = 1; i <= numCons; i++) {
        let rowHtml = `<div class="matrix-row">`;
        for (let j = 1; j <= numVars; j++) {
            rowHtml += `
                <input type="number" id="r${i}_x${j}" placeholder="a${i}${j}">
                <label>x<sub>${j}</sub></label>
                ${j < numVars ? ' + ' : ''}
            `;
        }

        rowHtml += `
            <select id="op_${i}" class="operator-select">
                <option value="<=">&le;</option>
                <option value=">=">&ge;</option>
                <option value="=">=</option>
            </select>
        `;

        rowHtml += `
            <input type="number" id="rhs_${i}" placeholder="b${i}">
        </div>`;
        
        matrixContainer.innerHTML += rowHtml;
    }
    modelArea.style.display = 'block';
}

// Función Principal Modificada para recibir el MÉTODO
async function iniciarCalculo(metodo) {

    // 1. Validación Estricta según el método [Req 5]
    if (!validarEntradas(metodo)) {
        return; // Si falla la validación, detiene todo.
    }

    // Recolectar datos
    const numVars = parseInt(document.getElementById('num-vars').value);
    const numCons = parseInt(document.getElementById('num-cons').value);
    const objType = document.getElementById('obj-type').value;

    let c = [];
    let A = [];
    let b = [];
    let operators = [];

    // Coeficientes de Z
    for (let i = 1; i <= numVars; i++) {
        c.push(Number(document.getElementById(`z_x${i}`).value));
    }

    // Matriz A, RHS y operadores
    for (let i = 1; i <= numCons; i++) {
        let row = [];
        for (let j = 1; j <= numVars; j++) {
            row.push(Number(document.getElementById(`r${i}_x${j}`).value));
        }
        A.push(row);
        b.push(Number(document.getElementById(`rhs_${i}`).value));

        let opVal = document.getElementById(`op_${i}`).value;
        operators.push(opVal);
    }

    // Enviar al Backend
    try {
        const response = await fetch('http://127.0.0.1:5000/api/solve_simplex', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                c: c,
                A: A,
                b: b,
                operators: operators,
                objective: objType,
                method: metodo // Método enviado para backend
            })
        });

        const result = await response.json();

        if (result.success) {

            // 🔥 NUEVA DETECCIÓN DE TIPOS DE RESPUESTA
            if (result.type === 'algebraic') {
                renderizarAlgebraico(result.steps);

            } else if (result.type === 'revised') {
                // 👇👇 NUEVO
                renderizarRevisado(result.steps);

            } else {
                // Simplex normal / 2 fases / m-grande
                renderizarTarjetas(result.steps);
            }

        } else {
            mostrarError("Error del sistema: " + result.error);
        }

    } catch (error) {
        mostrarError("¡El servidor no responde! ¿Está corriendo Python?");
    }
}


// --- VALIDACIÓN ROBUSTA (REQ 3.1 & REQ 5) ---
function validarEntradas(metodo) {
    const feedback = document.getElementById('feedback');
    feedback.style.display = 'none'; // Limpiar errores previos

    // 1. Validación Genérica: Campos Vacíos y Numéricos [Req 3.1]
    const inputs = document.querySelectorAll('input[type="number"]');
    for (let input of inputs) {
        if (input.value.trim() === '') {
            mostrarError("¡Koffing ataca! Hay campos vacíos en el modelo.");
            return false;
        }
    }

    // 2. Validación Genérica: RHS Negativo [Req 3.1]
    const numCons = parseInt(document.getElementById('num-cons').value);
    for(let i=1; i<=numCons; i++) {
        let rhs = parseFloat(document.getElementById(`rhs_${i}`).value);
        if (rhs < 0) {
            mostrarError(`¡Psyduck confundido! La restricción ${i} tiene un lado derecho negativo. Multiplica por -1 antes de ingresar.`);
            return false;
        }
    }

    // 3. Validación Específica: MÉTODO SIMPLEX ESTÁNDAR [Req 3.2.1]
    if (metodo === 'simplex') {
        // Regla A: Solo Maximización
        const objType = document.getElementById('obj-type').value;
        if (objType !== 'max') {
            mostrarError(" REGLA DE GIMNASIO: El Método Simplex Estándar solo acepta <strong>Maximización</strong>. Usa Minimización cambiando los signos o usa otro método.");
            return false;
        }

        // Regla B: Solo restricciones <=
        for(let i=1; i<=numCons; i++) {
            let op = document.getElementById(`op_${i}`).value;
            if (op !== '<=') {
                mostrarError(` MOVIMIENTO ILEGAL: El Simplex Estándar no soporta "${op}" en la restricción ${i}. <br><strong>Sugerencia:</strong> Usa el Método M-Grande o Dos Fases.`);
                return false;
            }
        }
    }

    // 4. Validación Específica: MÉTODO M GRANDE [Req 3.2.2]
    if (metodo === 'mgrande') {
        // M Grande soporta todo, pero verificamos que realmente sea necesario usar M Grande.
        // Si todas son <=, sugerimos Simplex (opcional, pero buena UX).
        let tieneArtificiales = false;
        for(let i=1; i<=numCons; i++) {
            let op = document.getElementById(`op_${i}`).value;
            if (op === '>=' || op === '=') tieneArtificiales = true;
        }
        
        // No es error, pero podríamos avisar. Por ahora lo dejamos pasar.
    }

    return true; // Todo correcto
}

function mostrarError(mensaje) {
    const feedback = document.getElementById('feedback');
    feedback.innerHTML = `
        <div style="display:flex; align-items:center; gap:10px;">
            <img src="https://img.pokemondb.net/sprites/red-blue/normal/koffing.png" width="50">
            <div><strong>ERROR DE ENTRENADOR:</strong><br>${mensaje}</div>
        </div>
    `;
    feedback.className = 'feedback-box error-msg';
    feedback.style.display = 'block';
}

function renderizarTarjetas(pasos) {
    const feedback = document.getElementById('feedback');
    feedback.style.display = 'block';
    feedback.innerHTML = ""; 

    pasos.forEach(paso => {
        // --- CORRECCIÓN CLAVE AQUÍ ---
        // Si es un mensaje de transición (Fase 1 -> Fase 2), mostrar banner y continuar
        if (paso.status === 'PHASE_TRANSITION') {
            feedback.innerHTML += `
                <div class="phase-banner">
                    <h3>${paso.phase}</h3>
                    <p>${paso.message}</p>
                </div>
            `;
            return; // ¡Saltar el resto del código para este paso específico!
        }
        // -----------------------------

        // Construcción normal de la tabla
        let tablaHTML = '<div class="matrix-container"><table>';
        
        // Headers (Aquí fallaba antes porque el paso de transición no tiene headers)
        if (paso.headers) {
            tablaHTML += '<thead><tr>';
            tablaHTML += '<th>Base</th>';
            paso.headers.forEach(h => { tablaHTML += `<th>${h}</th>`; });
            tablaHTML += '</tr></thead>';
        }

        // Body
        if (paso.tableau) {
            tablaHTML += '<tbody>';
            paso.tableau.forEach((fila, index) => {
                let rowName = (index < paso.tableau.length - 1) ? (paso.basic_vars[index] || `F${index}`) : "Z";
                let rowClass = (index === paso.pivot_row) ? 'highlight-row' : '';
                
                tablaHTML += `<tr class="${rowClass}">`;
                tablaHTML += `<td class="base-col"><strong>${rowName}</strong></td>`;
                fila.forEach((valor, colIndex) => {
                    let colClass = (colIndex === paso.pivot_col) ? 'highlight-col' : '';
                    let valStr = Number(valor).toFixed(2);
                    if (valStr.endsWith('.00')) valStr = parseInt(valor);
                    tablaHTML += `<td class="${colClass}">${valStr}</td>`;
                });
                tablaHTML += '</tr>';
            });
            tablaHTML += '</tbody></table></div>';
        }

        // Tarjeta
        // Definir color del encabezado según la fase
        let headerColor = '#FF0000'; // Rojo por defecto
        if (paso.phase === 'FASE 1') headerColor = '#A040A0'; // Morado
        if (paso.phase === 'FASE 2') headerColor = '#306567'; // Verde Azulado

        let html = `
        <div class="poke-card-report">
            <h3 class="card-header" style="background: ${headerColor}">
                ${paso.phase ? paso.phase + ' - ' : ''} Iteración ${paso.iteration}
            </h3>
            
            <div class="card-stats">
                <p><strong>Z = </strong> ${paso.z_value}</p>
                ${paso.entering_var ? `<p> Entra: <strong>${paso.entering_var}</strong></p>` : ''}
                ${paso.leaving_var ? `<p> Sale: <strong>${paso.leaving_var}</strong></p>` : ''}
            </div>

            ${tablaHTML}
            
            ${paso.status === 'OPTIMAL' ? '<div class="success-banner"> ¡SOLUCIÓN ÓPTIMA CAPTURADA! </div>' : ''}
            ${paso.status === 'INFEASIBLE' ? '<div class="error-banner"> PROBLEMA INFACTIBLE (W > 0)</div>' : ''}
            ${paso.status === 'UNBOUNDED' ? '<div class="error-banner"> SOLUCIÓN NO ACOTADA</div>' : ''}
        </div>
        `;
        feedback.innerHTML += html;
    });
}

function renderizarAlgebraico(soluciones) {
    const feedback = document.getElementById('feedback');
    feedback.style.display = 'block';
    
    if (soluciones.length === 0) {
        feedback.innerHTML = '<div class="error-banner"> No se encontraron soluciones básicas (Sistema Singular).</div>';
        return;
    }

    // Crear una gran tabla resumen
    let html = `
    <div class="poke-card-report" style="border-color: #F85888;">
        <h3 class="card-header" style="background: #F85888;">🔮 Soluciones Básicas Encontradas</h3>
        <div class="matrix-container">
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Variables Básicas</th>
                        <th>Valores (Punto)</th>
                        <th>Z</th>
                        <th>Estado</th>
                    </tr>
                </thead>
                <tbody>
    `;

    soluciones.forEach(sol => {
        let rowColor = '';
        let icon = '';
        
        if (sol.is_optimal) {
            rowColor = 'background-color: #d4edda;'; // Verde claro
            icon = 'optimo';
        } else if (sol.status === 'Factible') {
            rowColor = '';
            icon = 'Factible';
        } else {
            rowColor = 'background-color: #f8d7da; color: #721c24;'; // Rojo claro
            icon = 'infactible';
        }

        // Formatear valores
        let valStr = Object.entries(sol.values)
            .map(([k, v]) => `<strong>${k}</strong>=${v}`)
            .join(', ');

        html += `
            <tr style="${rowColor}">
                <td>${sol.id}</td>
                <td>{ ${sol.basic_indices.join(', ')} }</td>
                <td>${valStr}</td>
                <td><strong>${sol.z_value}</strong></td>
                <td>${icon} ${sol.status}</td>
            </tr>
        `;
    });

    html += `
                </tbody>
            </table>
        </div>
        <div style="text-align:center; padding:10px; font-style:italic;">
            "Alakazam ha analizado ${soluciones.length} futuros posibles."
        </div>
    </div>
    `;

    feedback.innerHTML = html;
}

function renderizarRevisado(pasos) {
    const feedback = document.getElementById('feedback');
    feedback.style.display = 'block';
    feedback.innerHTML = "";

    pasos.forEach(paso => {
        // Formatear Matriz Inversa B^-1 para HTML
        let bInvHTML = '<table class="matrix-small"><tbody>';
        paso.b_inv.forEach(row => {
            bInvHTML += '<tr>';
            row.forEach(val => {
                bInvHTML += `<td>${Number(val).toFixed(2)}</td>`;
            });
            bInvHTML += '</tr>';
        });
        bInvHTML += '</tbody></table>';

        // Formatear Multiplicadores Pi
        let piHTML = '[' + paso.pi_vector.map(v => Number(v).toFixed(2)).join(', ') + ']';

        // Formatear Variables Básicas y sus valores
        let solHTML = '<ul>';
        paso.basic_vars.forEach((v, i) => {
            let val = Number(paso.current_sol[i]).toFixed(2);
            solHTML += `<li><strong>${v}</strong> = ${val}</li>`;
        });
        solHTML += '</ul>';

        let html = `
        <div class="poke-card-report" style="border-color: #6a0dad;">
            <h3 class="card-header" style="background: #6a0dad;">Iteración ${paso.iteration}</h3>
            
            <div style="display:flex; flex-wrap:wrap; gap:20px; justify-content:center;">
                
                <div style="flex:1; min-width:200px;">
                    <h4>Variables Básicas ($X_B$)</h4>
                    ${solHTML}
                    <p><strong>Z Actual:</strong> ${paso.z_value}</p>
                </div>

                <div style="flex:1; min-width:200px;">
                    <h4>Matriz Base Inversa ($B^{-1}$)</h4>
                    ${bInvHTML}
                </div>

                <div style="flex:1; min-width:200px;">
                    <h4>Multiplicadores Simplex ($\pi$)</h4>
                    <div style="background:#eee; padding:5px; border-radius:5px; font-family:monospace;">
                        ${piHTML}
                    </div>
                </div>
            </div>

            <hr style="border-top: 1px dashed #6a0dad; margin: 15px 0;">

            <div class="card-stats">
                ${paso.entering_var ? `<p> Entra (Max Cj-Zj): <strong>${paso.entering_var}</strong></p>` : ''}
                ${paso.leaving_var ? `<p> Sale (Min Ratio): <strong>${paso.leaving_var}</strong></p>` : ''}
            </div>

            ${paso.status === 'OPTIMAL' ? '<div class="success-banner" style="background:#6a0dad;"> ÓPTIMO MATRICIAL ALCANZADO </div>' : ''}
        </div>
        `;
        feedback.innerHTML += html;
    });
}