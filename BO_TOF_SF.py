from simple_salesforce import Salesforce
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sf = Salesforce(username='colaborador@quintoandar.com',  #<---- correo de quitoandar
                password='contraseña',                   #<---- contrasñea 
                security_token='token_salesforce')       #<---- token de seguridad de salesforce

query_leads_history = sf.query_all(
'''
SELECT 
    LeadId,
    CreatedDate,
    Field,
    OldValue,
    NewValue
FROM LeadHistory
WHERE LeadId IN (SELECT Id FROM Lead WHERE Organizacion_de_venta_Texto__c LIKE '%Inmuebles 24%')
AND Field IN ('Status', 'Motivo_de_no_califica__c', 'Motivo_no_convierte__c') 
'''
)

query_leads_dim = sf.query_all(
'''
SELECT 
    Id,
    Ciudad_Texto__c,
    ContactId__c,
    ConvertedAccountId,
    ConvertedContactId,
    ConvertedOpportunityId,
    ConvertedDate,
    OwnerId,
    LeadSource,
    Provincia_Estado_Texto__c,
    Status,
    Motivo_de_no_califica__c,
    Motivo_no_convierte__c,
    CreatedDate,
    LastModifiedDate,
    Es_empresa__c
FROM Lead
WHERE Organizacion_de_venta_Texto__c LIKE '%Inmuebles 24%'
AND CreatedDate >= 2025-01-01T00:00:00Z
'''
)

query_propietarios = sf.query_all(
'''
SELECT 
    Id,
    Name
FROM User
WHERE Pais__c = 'Mx'
'''
)

query_oportunidades = sf.query_all(
'''
SELECT
    Id,
    AccountId,
    Amount,
    Status_de_la_OS__c,
    StageName,
    LastStageChangeDate,
    CloseDate,
    ContractId,
    CreatedDate,
    Fecha_inicio_de_vigencia__c,
    Fecha_fin_de_vigencia__c,
    Numero_OS__c,
    Numero_de_la_oportunidad__c,
    Status_vs_target__c
FROM Opportunity 
WHERE AccountId IN (
    SELECT 
        ConvertedAccountId
    FROM Lead
    WHERE Organizacion_de_venta_Texto__c LIKE '%Inmuebles 24%')
'''
)

teams = pd.read_csv('C:\\Users\\DiegoAntonioSalasTre\\Documents\\tof\\teams.csv')   #<---- archivo con los nombres de los ejecutivos comerciales y sus respectivos equipos    
                        
df_leads_history = pd.DataFrame(query_leads_history['records'])
df_leads_dim = pd.DataFrame(query_leads_dim['records'])
#df_leads_dim['ConvertedDate'] = df_leads_dim['ConvertedDate'].str.slice(0, 11)
df_propietarios = pd.DataFrame(query_propietarios['records'])
df_propietarios = pd.merge(df_propietarios, teams, left_on='Name', right_on='Ejecutivo')
df_oportunidades_dim = pd.DataFrame(query_oportunidades['records'])

df_leads_dim.drop('attributes', axis=1, inplace=True)
df_leads_history.drop('attributes', axis=1, inplace=True)
df_propietarios.drop('attributes', axis=1, inplace=True)
df_oportunidades_dim.drop('attributes', axis=1, inplace=True)

df_leads_history['CreatedDate'] = pd.to_datetime(df_leads_history['CreatedDate'])
df_leads_history['CreatedDate'] = df_leads_history['CreatedDate'].dt.tz_convert('America/Mexico_City')
df_leads_history['Periodo'] = df_leads_history['CreatedDate'].dt.to_period('M')
df_leads_history = df_leads_history[df_leads_history['Field'] == 'Status']

def treat_dates(df, cols):
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['Periodo'] = df['CreatedDate'].dt.to_period('M')

treat_dates(df_oportunidades_dim,['CreatedDate', 'LastStageChangeDate', 'CloseDate'])
#treat_dates(df_leads_dim,['CreatedDate', 'LastModifiedDate', 'ConvertedDate'])

df_leads_dim['CreatedDate'] = pd.to_datetime(df_leads_dim['CreatedDate'])
df_leads_dim['LastModifiedDate'] = pd.to_datetime(df_leads_dim['LastModifiedDate'])
df_leads_dim['ConvertedDate'] = pd.to_datetime(df_leads_dim['ConvertedDate'])
df_leads_dim['ConvertedDate'] = df_leads_dim['ConvertedDate'].dt.tz_localize('UTC')


df_leads_dim['CreatedDate'] = df_leads_dim['CreatedDate'].dt.tz_convert('America/Mexico_City')
df_leads_dim['LastModifiedDate'] = df_leads_dim['LastModifiedDate'].dt.tz_convert('America/Mexico_City')
df_leads_dim['ConvertedDate'] = df_leads_dim['ConvertedDate'].dt.tz_convert('America/Mexico_City')

df_leads_dim['Periodo'] = df_leads_dim['CreatedDate'].dt.to_period('M')

#df_leads_dim = df_leads_dim[df_leads_dim['Es_empresa__c'] != 'Personal']
df_leads_dim = df_leads_dim[df_leads_dim['Periodo'] >= '2025-01']
df_oportunidades_dim = df_oportunidades_dim[df_oportunidades_dim['Periodo'] >= '2025-01']

valores_a_no_conv = ['No califica', 'Califica pero no interesado']
df_leads_history['NewValue'] = df_leads_history['NewValue'].replace(valores_a_no_conv, 'No convierte')
df_leads_history['OldValue'] = df_leads_history['OldValue'].replace(valores_a_no_conv, 'No convierte')

valores_a_contactando = ['Contacting', 'Trabajando']
df_leads_history['NewValue'] = df_leads_history['NewValue'].replace(valores_a_contactando, 'Contactando')
df_leads_history['OldValue'] = df_leads_history['OldValue'].replace(valores_a_contactando, 'Contactando')

df_leads_history = df_leads_history.sort_values(
    by = ['LeadId', 'CreatedDate'],
    ascending = [True, True])

etapas_leads = ['Nuevo', 'No contactado', 'Contactando', 'Negociación', 'Convertido', 'No convierte']

df_leads_history['NewValue'] = pd.Categorical(
    df_leads_history['NewValue'], 
    categories=etapas_leads, 
    ordered=True
)

df_leads_history['OldValue'] = pd.Categorical(
    df_leads_history['OldValue'], 
    categories=etapas_leads, 
    ordered=True
)

df_leads_dim['Status'] = pd.Categorical(
    df_leads_dim['Status'], 
    categories=etapas_leads, 
    ordered=True
)

df_leads_history['Ciclos'] = 0 

def contar_ciclos(grupo):
    ciclo = 1
    ciclos = []
    last_terminal = None
    for i, row in grupo.iterrows():
        if row['NewValue'] == 'Nuevo' and last_terminal is not None:
            ciclo += 1
        if row['NewValue'] in ['Convertido','No convierte']:
            last_terminal = row['NewValue']
        ciclos.append(ciclo)
    grupo['Ciclos'] = ciclos
    return grupo

df_leads_history = df_leads_history.groupby('LeadId').apply(contar_ciclos).reset_index(drop = True)

serie_propietarios = df_propietarios.set_index('Id')['Name']
serie_equipos = df_propietarios.set_index('Id')['Equipo']

df_leads_dim['Propietario'] = df_leads_dim['OwnerId'].map(serie_propietarios).fillna('Desconocido')
df_leads_dim['Equipo'] = df_leads_dim['OwnerId'].map(serie_equipos).fillna('Desconocido')
df_leads_dim = df_leads_dim[df_leads_dim['Equipo'] != 'Desconocido']

serie_ejecutivos = df_leads_dim.set_index('Id')['Equipo']
serie_propietario = df_leads_dim.set_index('Id')['Propietario']

df_leads_history['Equipo'] = df_leads_history['LeadId'].map(serie_ejecutivos).fillna('Desconocido')
df_leads_history['Propietario'] = df_leads_history['LeadId'].map(serie_propietario).fillna('Desconocido')
df_leads_history = df_leads_history[df_leads_history['Equipo'] != 'Desconocido']

ultima_etapa_por_ciclo = df_leads_history.sort_values('CreatedDate').groupby(['LeadId','Ciclos']).tail(1)

ultima_etapa_por_lead = ultima_etapa_por_ciclo.groupby('LeadId').tail(1)
conversion_leads = ultima_etapa_por_lead['NewValue'].value_counts(normalize=True) * 100

conversion_ciclos = ultima_etapa_por_ciclo['NewValue'].value_counts(normalize=True) * 100

transitions = df_leads_history.dropna(subset=['OldValue'])
filtrado_transitions = transitions[transitions['OldValue'] != transitions['NewValue']]
transition_matrix = filtrado_transitions.groupby(['OldValue','NewValue']).size().unstack(fill_value=0)

plt.figure(figsize=(8,6))
ax = sns.heatmap(transition_matrix, 
            annot=True, 
            fmt=',.0f', 
            cmap='Blues')
ax.set(xlabel='Estado Nuevo',
       ylabel='Estado Viejo')
plt.title('Matriz de transiciones entre etapas')
plt.show()

ultima_etapa_por_ciclo.to_csv('C:\\Users\\DiegoAntonioSalasTre\\Documents\\tof\\nivel_ciclos.csv', index=False)
df_leads_dim['Equipo'] = df_leads_dim['Equipo'].astype('category')
serie_creacion = df_leads_dim.set_index('Id')['Periodo']

ultima_etapa_por_ciclo['Cohort'] = ultima_etapa_por_ciclo['LeadId'].map(serie_creacion)
cohort_total = df_leads_dim.groupby('Periodo').size().rename('Nuevos')
filtro_growth = df_leads_dim[(df_leads_dim['Equipo'] == 'Growth') & (df_leads_dim['Status'] == 'Convertido')]
cohort_total_growth = filtro_growth.groupby('Periodo').size().rename('Nuevos')

filtro_convertidos = df_leads_dim[df_leads_dim['Status'] == 'Convertido']
filtro_convertidos['CreatedDate'] = pd.to_datetime(filtro_convertidos['CreatedDate'], errors='coerce')
filtro_convertidos['LastModifiedDate'] = pd.to_datetime(filtro_convertidos['LastModifiedDate'], errors='coerce')
filtro_convertidos['ConvertedDate'] = pd.to_datetime(filtro_convertidos['ConvertedDate'], errors='coerce')

mask = filtro_convertidos['ConvertedDate'].isna()
filtro_convertidos.loc[mask, 'ConvertedDate'] = filtro_convertidos.loc[mask, 'LastModifiedDate']

filtro_convertidos['PeriodoConvertido'] = filtro_convertidos['ConvertedDate'].dt.to_period('M')
filtro_convertidos['MesRelativo'] = ((filtro_convertidos['ConvertedDate'].dt.to_period('M') - filtro_convertidos['Periodo']).apply(lambda x: x.n) + 1)

filtro_convertidos.to_csv('C:\\Users\\DiegoAntonioSalasTre\\Documents\\tof\\convertidos_data.csv', index=False)

pivote_convertidos = pd.pivot_table(data=filtro_convertidos, index='Periodo', columns='PeriodoConvertido', values='Id', dropna=True, aggfunc='size')
pivote_convertidos['Total'] = pivote_convertidos.sum(axis=1)

pivote_cohorts = pivote_convertidos.reset_index()

pivote_cohorts.to_csv('C:\\Users\\DiegoAntonioSalasTre\\Documents\\tof\\pivot_cohorts.csv', index=False)

cohort_counts = filtro_convertidos.groupby(['Periodo', 'MesRelativo', 'LeadSource', 'Propietario', 'Equipo']).size().reset_index(name='Cantidad')
cohort_counts = cohort_counts[cohort_counts['Cantidad'] >= 1]
cohort_pivot = cohort_counts.pivot_table(index='Periodo', columns='MesRelativo', values='Cantidad', aggfunc='sum', fill_value=0)
cohort_counts.to_csv('C:\\Users\\DiegoAntonioSalasTre\\Documents\\tof\\resumen_mes_relativo.csv', index=False)
cohort_pivot.columns = [f'M{c}' for c in cohort_pivot.columns]
cohort_pivot = cohort_pivot.fillna(0).astype(int)
cohort_pivot['Total'] = cohort_pivot.sum(axis=1)
cohort_pivot = cohort_pivot.reset_index()

cohort_pivot['Tamaño cohorte'] = cohort_pivot['Periodo'].map(cohort_total).fillna('Desconocido')

pivote_doc = cohort_pivot[['Periodo', 'Tamaño cohorte', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']]

pivote_doc.to_csv('C:\\Users\\DiegoAntonioSalasTre\\Documents\\tof\\pivote_cohortes.csv', index=False)

cohort_convertidos = (
    df_leads_dim[df_leads_dim['Status'] == 'Convertido']
    .groupby('Periodo')
    .size()
    .rename('Convertidos')
)

cohort_resumen = pd.concat([cohort_total, cohort_convertidos], axis=1).fillna(0).reset_index()

cohort_resumen['Convertidos'] = cohort_resumen['Convertidos'].astype(int)
cohort_resumen['Conversión'] = cohort_resumen['Convertidos'] / cohort_resumen['Nuevos']

resumen = df_leads_dim.groupby(['Periodo', 'Status', 'Equipo', 'Propietario', 'LeadSource']).size().rename('Prospectos').reset_index()

resumen = resumen[resumen['Prospectos'] >= 1]

resumen.to_csv('C:\\Users\\DiegoAntonioSalasTre\\Documents\\tof\\resumen_leads_cohort.csv', index=False)

filtro_no_convierte = df_leads_dim[df_leads_dim['Status'] == 'No convierte'] 

no_convierte = filtro_no_convierte.groupby(['Periodo', 'Status', 'Equipo', 'Propietario', 'LeadSource', 'Motivo_no_convierte__c']).size().rename('Prospectos').reset_index()
no_convierte = no_convierte[no_convierte['Prospectos'] >= 1]

no_convierte.to_csv('C:\\Users\\DiegoAntonioSalasTre\\Documents\\tof\\resumen_leads_cohort_descartes.csv', index=False)

pivot_descartes = pd.pivot_table(data=no_convierte, index='Motivo_no_convierte__c', columns='Periodo', values='Prospectos', aggfunc='sum', fill_value=0)

cohort_total_equipos = df_leads_dim.groupby(['Periodo', 'Equipo']).size().rename('Nuevos')

cohort_convertidos_equipos = (
    df_leads_dim[df_leads_dim['Status'] == 'Convertido']
    .groupby(['Periodo', 'Equipo'])
    .size()
    .rename('Convertidos')
)

cohort_resumen_equipos = pd.concat([cohort_total_equipos, cohort_convertidos_equipos], axis=1).fillna(0).reset_index()

sns.set_theme(style='whitegrid')

cohort_resumen_equipos['Periodo_dt'] = cohort_resumen_equipos['Periodo'].dt.to_timestamp()
cohort_resumen['Periodo_dt'] = cohort_resumen['Periodo'].dt.to_timestamp()
cohort_resumen_equipos['Conversión'] = cohort_resumen_equipos['Convertidos'] / cohort_resumen_equipos['Nuevos']

plt.figure(figsize=(10,6))
ax2 = sns.lineplot(
    data=cohort_resumen,
    x='Periodo_dt',
    y='Conversión',
    marker='o'     
)

plt.title('Conversión por periodo')
plt.xlabel('Periodo')
plt.ylabel('% de conversión')
plt.xticks(rotation=45)  
plt.tight_layout()
plt.show()

def format_k(x, pos):
    return f'{x/1000:.0f}K'

def to_percent(y, pos):
    return f'{y*100:.0f}%'

cohort_resumen['Periodo_dt'] = pd.to_datetime(cohort_resumen['Periodo_dt'])
cohort_resumen['Periodo_str'] = cohort_resumen['Periodo_dt'].dt.strftime('%Y-%m')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

sns.barplot(
    data=cohort_resumen,
    x='Periodo_str',
    y='Nuevos',
    ax=ax1,
    color='skyblue'
)
for p in ax1.patches:
    height = p.get_height()
    ax1.text(
        x=p.get_x() + p.get_width()/2,
        y=height + max(cohort_resumen['Nuevos'])*0.01,
        s=f'{height/1000:.1f}K',
        ha='center'
    )
ax1.yaxis.set_major_formatter(FuncFormatter(format_k))
ax1.set_ylabel('Tamaño del cohorte')
ax1.set_xlabel('')
ax1.grid(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
#ax1.spines['bottom'].set_visible(False)

sns.lineplot(
    data=cohort_resumen,
    x='Periodo_str',
    y='Conversión',
    marker='o',
    ax=ax2
)

for x, y in zip(range(len(cohort_resumen['Periodo_str'])), cohort_resumen['Conversión']):
    ax2.text(
        x=x,
        y=y + max(cohort_resumen['Conversión'])*0.01,
        s=f'{y*100:.0f}%',
        ha='center'
    )
    
ax2.yaxis.set_major_formatter(FuncFormatter(to_percent))
ax2.set_ylabel('Tasa de conversión')
ax2.set_xlabel('Periodo')
ax2.grid(False, axis='y')

plt.setp(ax2.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.show()

cerradas_ganadas = df_oportunidades_dim[df_oportunidades_dim['StageName'] == 'Cerrada Ganada']
cerradas_perdidas = df_oportunidades_dim[df_oportunidades_dim['StageName'] == 'Cerrada Perdida']
oportunidades_creadas = df_oportunidades_dim.groupby('AccountId')['Id'].size()
oportunidades_ganadas = cerradas_ganadas.groupby('AccountId')['Id'].size()
oportunidades_perdidas = cerradas_perdidas.groupby('AccountId')['Id'].size()

etapas_oportunidad = [
    'Cerrada Perdida',
    'Descubrimiento',
    'Inicial',
    'Negociación',
    'Presupuesto Enviado',
    'Generar Orden de Servicio',
    'Generar Solicitud de Pago',
    'Cerrada Ganada'
]

for stage in etapas_oportunidad:
    tipo = df_oportunidades_dim[df_oportunidades_dim['StageName'] == stage]
    totales = tipo.groupby('AccountId')['Id'].size()
    df_leads_dim[stage] = df_leads_dim['ConvertedAccountId'].map(totales).fillna(0) > 0

etapas_rank = {etapa: i for i, etapa in enumerate(etapas_oportunidad)}

def obtener_etapa(row):
    for etapa in reversed(etapas_oportunidad):
        if row[etapa] == True:
            return etapa
    return None 

df_leads_dim['Etapa_final'] = df_leads_dim.apply(obtener_etapa, axis=1)

resumen_oportunidades = df_leads_dim.groupby(['Periodo', 'Etapa_final', 'Equipo', 'Propietario', 'LeadSource'])['Id'].size().reset_index(name='Prospectos_unicos')
resumen_oportunidades = resumen_oportunidades[resumen_oportunidades['Prospectos_unicos'] >= 1]

resumen_oportunidades.to_csv('C:\\Users\\DiegoAntonioSalasTre\\Documents\\tof\\data_oportunidades.csv', index=False)














