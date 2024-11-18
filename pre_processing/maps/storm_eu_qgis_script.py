"""
Model exported as python.
Name : storm_eu_square_modelleur
Group : 
With QGIS : 33411
"""

import sys
#sys.path.append('/Applications/QGIS-LTR.app/Contents/Resources/python')

from qgis.core import QgsProcessing
from qgis.core import QgsProcessingAlgorithm
from qgis.core import QgsProcessingMultiStepFeedback
from qgis.core import QgsProcessingParameterVectorLayer
from qgis.core import QgsProcessingParameterFileDestination
from qgis.core import QgsProcessingParameterFeatureSink
from qgis.core import QgsCoordinateReferenceSystem
from qgis.core import QgsExpression
#import processing


class Storm_eu_square_modelleur(QgsProcessingAlgorithm):

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer('limite_eu', 'limite_eu', types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterVectorLayer('tracks', 'tracks', types=[QgsProcessing.TypeVectorPoint], defaultValue=None))
        self.addParameter(QgsProcessingParameterFileDestination('Square_eu_csv_qgis_friendly', 'square_eu_csv_QGIS_friendly', fileFilter='Microsoft Excel (*.xlsx);;Open Document Spreadsheet (*.ods)', createByDefault=True, defaultValue=''))
        self.addParameter(QgsProcessingParameterFileDestination('Square_eu_csv', 'square_eu_csv', fileFilter='Microsoft Excel (*.xlsx);;Open Document Spreadsheet (*.ods)', createByDefault=True, defaultValue=None))
        self.addParameter(QgsProcessingParameterFeatureSink('Square_eu_', 'square_eu_', type=QgsProcessing.TypeVectorAnyGeometry, createByDefault=True, defaultValue='TEMPORARY_OUTPUT'))

    def processAlgorithm(self, parameters, context, model_feedback):
        # Use a multi-step feedback, so that individual child algorithm progress reports are adjusted for the
        # overall progress through the model
        feedback = QgsProcessingMultiStepFeedback(9, model_feedback)
        results = {}
        outputs = {}

        # Créer une couche de points à partir d'une table
        alg_params = {
            'INPUT': parameters['tracks'],
            'MFIELD': '',
            'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:4326'),
            'XFIELD': 'lon_west',
            'YFIELD': 'lat_south',
            'ZFIELD': '',
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['CrerUneCoucheDePointsPartirDuneTable'] = processing.run('native:createpointslayerfromtable', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        # Calculatrice de champ
        alg_params = {
            'FIELD_LENGTH': 3,
            'FIELD_NAME': 'step',
            'FIELD_PRECISION': 0,
            'FIELD_TYPE': 1,  # Entier (32bit)
            'FORMULA': '@id-1',
            'INPUT': outputs['CrerUneCoucheDePointsPartirDuneTable']['OUTPUT'],
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['CalculatriceDeChamp'] = processing.run('native:fieldcalculator', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        # Translation (de + 4°)
        alg_params = {
            'DELTA_M': 0,
            'DELTA_X': 4,
            'DELTA_Y': 4,
            'DELTA_Z': 0,
            'INPUT': outputs['CalculatriceDeChamp']['OUTPUT'],
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['TranslationDe4'] = processing.run('native:translategeometry', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}

        # Tampon (carré de 4°)
        alg_params = {
            'DISSOLVE': False,
            'DISTANCE': 4,
            'END_CAP_STYLE': 2,  # Carré
            'INPUT': outputs['TranslationDe4']['OUTPUT'],
            'JOIN_STYLE': 0,  # Rond
            'MITER_LIMIT': 2,
            'SEGMENTS': 5,
            'SEPARATE_DISJOINT': False,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['TamponCarrDe4'] = processing.run('native:buffer', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(4)
        if feedback.isCanceled():
            return {}

        # Extraire par localisation
        alg_params = {
            'INPUT': outputs['TamponCarrDe4']['OUTPUT'],
            'INTERSECT': parameters['limite_eu'],
            'PREDICATE': [0],  # intersecte
            'OUTPUT': parameters['Square_eu_']
        }
        outputs['ExtraireParLocalisation'] = processing.run('native:extractbylocation', alg_params, context=context, feedback=feedback, is_child_algorithm=True)
        results['Square_eu_'] = outputs['ExtraireParLocalisation']['OUTPUT']

        feedback.setCurrentStep(5)
        if feedback.isCanceled():
            return {}

        # Calculatrice de champ lon_west
        alg_params = {
            'FIELD_LENGTH': 3,
            'FIELD_NAME': QgsExpression("'lon_west'").evaluate(),
            'FIELD_PRECISION': 6,
            'FIELD_TYPE': 0,  # Décimal (double)
            'FORMULA': 'round(lon_west + 360,6)',
            'INPUT': outputs['ExtraireParLocalisation']['OUTPUT'],
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['CalculatriceDeChampLon_west'] = processing.run('native:fieldcalculator', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(6)
        if feedback.isCanceled():
            return {}

        # Calculatrice de champ lon_east
        alg_params = {
            'FIELD_LENGTH': 3,
            'FIELD_NAME': QgsExpression("'lon_east'").evaluate(),
            'FIELD_PRECISION': 6,
            'FIELD_TYPE': 0,  # Décimal (double)
            'FORMULA': 'round(lon_east + 360,6)',
            'INPUT': outputs['CalculatriceDeChampLon_west']['OUTPUT'],
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['CalculatriceDeChampLon_east'] = processing.run('native:fieldcalculator', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(7)
        if feedback.isCanceled():
            return {}

        # Exporter vers un tableur
        alg_params = {
            'FORMATTED_VALUES': False,
            'LAYERS': outputs['ExtraireParLocalisation']['OUTPUT'],
            'OVERWRITE': True,
            'USE_ALIAS': False,
            'OUTPUT': parameters['Square_eu_csv_qgis_friendly']
        }
        outputs['ExporterVersUnTableur'] = processing.run('native:exporttospreadsheet', alg_params, context=context, feedback=feedback, is_child_algorithm=True)
        results['Square_eu_csv_qgis_friendly'] = outputs['ExporterVersUnTableur']['OUTPUT']

        feedback.setCurrentStep(8)
        if feedback.isCanceled():
            return {}

        # Exporter vers un tableur
        alg_params = {
            'FORMATTED_VALUES': False,
            'LAYERS': outputs['CalculatriceDeChampLon_east']['OUTPUT'],
            'OVERWRITE': True,
            'USE_ALIAS': False,
            'OUTPUT': parameters['Square_eu_csv']
        }
        outputs['ExporterVersUnTableur'] = processing.run('native:exporttospreadsheet', alg_params, context=context, feedback=feedback, is_child_algorithm=True)
        results['Square_eu_csv'] = outputs['ExporterVersUnTableur']['OUTPUT']
        return results

    def name(self):
        return 'storm_eu_square_modelleur'

    def displayName(self):
        return 'storm_eu_square_modelleur'

    def group(self):
        return ''

    def groupId(self):
        return ''

    def createInstance(self):
        return Storm_eu_square_modelleur()
