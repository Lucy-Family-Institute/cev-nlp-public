from typeguard import typechecked
import warnings

short_dict = [
    ("Impactos Comunitarios o Colectivos", "[ImpColectivos]"),
    ("Impactos Comunitarios", "[ImpColectivos]"),
    ("Impactos", "[I]"),
    ("Transfronterizas", "[Tf]"),
    ("N3 Actores", "[AA]"),
    ("Políticas públicas", "[PP]"),
    ("Hechos -", "[HV] -"),
    ("Entidades", "[Ent]"),
    ("Organizaciones", "[Org]"),
    ("Impact Emoc Salud Mental Fisica", "[SMF]"),
    ("Impacto Democracia", "[ImpDemo]"),
    ("Afectación a la Participación ciudadana", "[AfecParticipCiudadana]"),
    ("Impacto Familiares", "[ImpFami]"),
    ("N7 Pueblos étnicos", "[N7]"),
    ("Pérdida de autonomía y dificultad en la toma de decisiones", "[PérdidaAutonom]"),
    ("N42 Despojo", "[N42]"),
    ("Afrontamientos", "[A]"),
    ("Estigmatización", "[Es]"),
    ("Ilegales", "[Il]"),
    ("N2 Estado", "[N2]"),
    ("N1 Democracia", "[N1]"),
    ("Causas del desplazamiento/abandono/confinamiento", "[Causas]"),
    ("Desplazamiento forzado", "[DForzado]"),
    ("Ejecución Extrajudicial Arbitraria", "[EEA]"),
    ("Dinámicas Espaciales Actores Armados", "[DinamEspacAA]"),
    ("Salud", "[S]"),
    ("Actividades sanitarias", "[AS]"),
    ("Infracciones a la misión médica", "[InfraccionesMM]"),
    ("Infracciones contra la actividad sanitaria", "[ActividadSanitaria]"),
    ("Cooptación de recursos de salud por actores armados", "[CooptaciónRecursosAA]"),
    ("Infracciones contra bienes protegidos en salud", "[BienesProtegidos]"),
    ("Salud sexual y reproductiva", "[SSR]"),
    ("Atención sanitaria a combatientes de grupos armados", "[AtenciónCombatientesAA]"),
    ("Resistencia", "[R]"),
    ("Proceso de paz", "[PPaz]"),
    ("Políticas Internacionales", "[PI]"),
    ("Divipola/Sitios/regiones", "[Lugar]"),
    ("N41 Economía", "[N41]"),
    ("Vida campesina y conflicto armado", "[VidaCampesina]"),
    ("No Repetición", "[NR]"),
    ("R Estado", "[Estado]"),
    ("R Reintegración excombatientes", "[ReintegraciónExcombatientes]"),
    ("R Drogas ilícitas narcotráfico", "[Narcotráfico]"),
    ("R Educación", "[Educación]"),
    ("R DESCA", "[DESCA]"),
    ("N82 Dim. Internacionales", "[N82]"),
    ("Apoyo militar actores armados", "[ApoyoMilitarAA]"),
    ("Apoyo político actores conflicto", "[ApoyoPolíticoAC]"),
    ("N81 Exilio", "[N81]"),
    ("Afrontamientos individuales", "[AfrontIndiv]"),
    ("Relaciones cultura de acogida", "[CulturaAcogida]"),
    ("Estatus migratorio", "[EMigratorio]"),
    ("Incremento de las conflictividades; violencias y/o deshumanización de la vida", "[IncrementoViolencia]"),
    ("Control a la poblacion civil", "[ControlPobCivil]"),
    ("Regulación de la vida social y comunitaria", "[RegVidaSocial]"),
    ("Prácticas de atención a combatientes heridos o enfermos por parte del personal sanitario de los grupos armados", "[AtenciónPorPersSanitarioAA]"),
    ("Retorno y reasentamiento en zona rural", "[RetornoZonaRural]"),
    ("Medios y participantes que actúan en el despojo", "[ParticipantesDespojo]"),
    ("Revictimización en el proceso de reclamación de tierras o retorno", "[RevicReclamTierra/Retorno]"),
    ("Dinámicas urbanas del despojo y el desplazamiento", "[UrbanoDespojoDesplazam]"),
    ("Violencia Política y represión a la protesta social", "[ViolenPolitica&RepresionProtesta]"),
    ("Pérdida o imposibilidad de acceder a la Seguridad social: pensión y/o salud", "[PérdidaPensión/Salud]"),
    ("étnico-racionales", "etnico-raciales"),
]

def corta(etiqueta):
    warnings.warn("Function `corta` is deprecated. Change your code to use `abbreviate` instead!", DeprecationWarning)
    return abbreviate(etiqueta)

@typechecked
def abbreviate(label: str) -> str:
    """
    Abbreviates an analytical or named label. Named entity labels are
    also desambiguated.

    Parameters
    ----------
    label: str
        A label from the label tree (only the type). If it is an
        entity, it is the entity type, joined through ' - ' with
        the raw name of the entity.

    Returns
    -------
    str
        The label either abbreviated or desambiguated.
    """
    for key, value in short_dict:
        if key in label:
            label = label.replace(key, value)
    if "Entidades" in label:
        pass
    return label

@typechecked
def desambiguar_entidad(entidad: str) -> str:
    """
    Desambigua una entidad nombrada.

    Parametros
    ----------
    entidad: str
        Una entidad nombrada. La primera parte es el tipo de
        entidad nombrada, y la segunda parte contiene el texto
        anotado de esa entidad. Se desambigua esa segunda parte.

    Retorna
    -------
    str
        La etiqueta con la parte del texto de la entidad desambiguada.
    """
    # TODO: create disambiguation dictionary
    return None
