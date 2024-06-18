from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

API_key = 'I6eCsG3fVH6k-L_4MYAbvgD_eoXk1U480UC8GezkUUcc'
url = 'https://api.eu-gb.language-translator.watson.cloud.ibm.com/instances/209d939b-59cb-4001-a707-013244cee735'

authenticator = IAMAuthenticator(apikey=API_key)

langtranslator = LanguageTranslatorV3(version='2018-05-01', authenticator=authenticator)

authenticator = IAMAuthenticator(apikey=API_key)

langtranslator = LanguageTranslatorV3(version='2018-05-01', authenticator=authenticator)


langtranslator.set_service_url(url)

translation = langtranslator.translate(text='Hello World',model_id='en-pt')

print(translation.get_result())

text = 'According to consensus in modern genetics anatomically modern humans first arrived on the Indian subcontinent from Africa between 73,000 and 55,000 years ago.[1] However, the earliest known human remains in South Asia date to 30,000 years ago. Settled life, which involves the transition from foraging to farming and pastoralism, began in South Asia around 7,000 BCE. At the site of Mehrgarh presence can be documented of the domestication of wheat and barley, rapidly followed by that of goats, sheep, and cattle.[2] By 4,500 BCE, settled life had spread more widely,[2] and began to gradually evolve into the Indus Valley Civilization, an early civilization of the Old world, which was contemporaneous with Ancient Egypt and Mesopotamia. This civilisation flourished between 2,500 BCE and 1900 BCE in what today is Pakistan and north-western India, and was noted for its urban planning, baked brick houses, elaborate drainage, and water supply.[3]'

translation = langtranslator.translate(text=text,model_id='en-pt')

print(translation.get_result())

print(translation.get_result()['translations'][0]['translation'])