#################################################################
#                                                               #
#    Urdu-Sanskrit synonyms dictionary with writing variants    #
#                      Version 382 words                        #
#                                                               #
#################################################################

import pandas as pd
from numpy import nan
from prettytable import PrettyTable

# ANSI color codes for terminal output
CYAN = '\033[1;36m'
YELLOW = '\033[1;33m'
WHITE = '\033[1;37m'
MAGENTA = '\033[1;35m'
GREEN = '\033[1;32m'
BLUE = '\033[1;34m'
ORANGE = '\033[38;5;208m'
RED = '\033[1;31m'
RESET = '\033[0m'

# Color assignments for word origins
PA_COLOR = GREEN      # Perso-Arabic words in green
SKRT_COLOR = ORANGE   # Sanskrit words in orange


class RawVocabulary:
    """
    A class for managing Urdu-Sanskrit synonym pairs with their linguistic properties.

    This class provides functionality to print(mapping_dict), analyze, and process a vocabulary
    of synonym pairs between Perso-Arabic (PA) and Sanskrit (SKRT) words in Hindi/Urdu.
    It contains 382 carefully curated synonym pairs for linguistic analysis.
    
    The vocabulary structure stores each word with:
    - Meaning: English translation/definition
    - Origin: 'PA' (Perso-Arabic) or 'SKRT' (Sanskrit)
    - POS: Part of speech (noun, adj, adv, ppos, etc.)
    
    Key Features:
    - Statistical analysis of vocabulary composition
    - Conversion to pandas DataFrame for data manipulation
    - Word embeddings integration using Word2Vec models
    - Similarity analysis between synonym pairs
    - Frequency difference calculations
    
    Usage:
        vocab = RawVocabulary()
        vocab.display_stats()  # Show vocabulary statistics
        df = vocab.convert_to_df()  # Convert to DataFrame
        
    Attributes:
        synonym_list (Dict): Dictionary containing all synonym pairs with their properties
    """

    synonym_list = {
        'मुताबिक़': ['according to', 'PA', 'ppos'],
        'अनुसार': ['according to', 'SKRT', 'ppos'],
        
        'फ़ायदा': ['advantage, benefit', 'PA', 'noun'],
        'लाभ': ['advantage, benefit', 'SKRT', 'noun'],
        
        'सलाह': ['advice', 'PA', 'noun'],
        'सम्मति': ['advice', 'SKRT', 'noun'],
        
        'ख़िलाफ़': ['against', 'PA', 'ppos'],
        'विरोध': ['against', 'SKRT', 'ppos'],
        
        'क़बूल': ['agreement', 'PA', 'noun'],
        'सहमति': ['agreement', 'SKRT', 'noun'],
        
        'हवा': ['air', 'PA', 'noun'],
        'वायु': ['air', 'SKRT', 'noun'],
        
        'हमेशा': ['always', 'PA', 'adv'],
        'सदा': ['always', 'SKRT', 'adv'],
        
        'और': ['and', 'PA', 'adv'],
        'तथा': ['and', 'SKRT', 'adv'],
        
        'इलाक़ा': ['area', 'PA', 'noun'],
        'क्षेत्र': ['area', 'SKRT', 'noun'],
        
        'फ़ौज': ['army', 'PA', 'noun'],
        'सेना': ['army', 'SKRT', 'noun'],
        
        'नफ़रत': ['aversion', 'PA', 'noun'],
        'घृणा': ['aversion', 'SKRT', 'noun'],
        
        'ख़ूबसूरत': ['beautiful', 'PA', 'adj'],
        'सुंदर': ['beautiful', 'SKRT', 'adj'],
        
        'आगाज़': ['beginning', 'PA', 'noun'],
        'शुरुआत': ['beginning', 'SKRT', 'noun'],
        
        'किताब': ['book', 'PA', 'noun'],
        'पुस्तक': ['book', 'SKRT', 'noun'],
        
        'सरहद': ['border', 'PA', 'noun'],
        'सीमा': ['border', 'SKRT', 'noun'],
        
        'ख़ून': ['blood', 'PA', 'noun'],
        'रक्त': ['blood', 'SKRT', 'noun'],
        
        'मसरूफ़': ['busy', 'PA', 'adj'],
        'व्यस्त': ['busy', 'SKRT', 'adj'],    
        
        'लेकिन': ['but', 'PA', 'adv'],
        'किंतु': ['but', 'SKRT', 'adv'],
        
        'वजह': ['cause, reason', 'PA', 'noun'],
        'कारण': ['cause, reason', 'SKRT', 'noun'],
        
        #'बदल': ['change', 'PA', 'noun'],
        #'परिवर्तन': ['change', 'SKRT', 'noun'],                   # UNDERREPRESENTED WORD IN THE CONTROL CORPUS
        
        'शहर': ['city', 'PA', 'noun'],
        'नगर': ['city', 'SKRT', 'noun'],
        
        'जमात': ['class (school)', 'PA', 'noun'],
        'कक्षा': ['class (school)', 'SKRT', 'noun'],
        
        'ज़ाहिर': ['clear, evident', 'PA', 'adj'],
        'स्पष्ट': ['clear, evident', 'SKRT', 'adj'],
        
        'जुर्म': ['crime', 'PA', 'noun'],
        'अपराध': ['crime', 'SKRT', 'noun'],
        
        'मुक़ाबले': ['comparison (in)', 'PA', 'ppos'],
        'तुलना': ['comparison (in)', 'SKRT', 'ppos'],
        
        'मुसलसल': ['continuously', 'PA', 'adv'],
        'लगातार': ['continuously', 'SKRT', 'adv'],
        
        'लाश': ['corpse', 'PA', 'noun'],
        'शव': ['corpse', 'SKRT', 'noun'],
        
        'मुल्क': ['country', 'PA', 'noun'],
        'देश': ['country', 'SKRT', 'noun'],
        
        'तारीख़': ['date', 'PA', 'noun'],
        'तिथि': ['date', 'SKRT', 'noun'],
        
        'महबूबा': ['dear, beloved', 'PA', 'noun'],
        'प्यारा': ['dear, beloved', 'SKRT', 'noun'],
        
        'मौत': ['death', 'PA', 'noun'],
        'मृत्यु': ['death', 'SKRT', 'noun'],
        
        'फ़ैसला': ['decision', 'PA', 'noun'],
        'निर्णय': ['decision', 'SKRT', 'noun'],
        
        'एलान': ['declaration', 'PA', 'noun'],
        'घोषणा': ['declaration', 'SKRT', 'noun'],
        
        'शिकस्त': ['defeat', 'PA', 'noun'],
        'पराजय': ['defeat', 'SKRT', 'noun'],
        
        'कमी': ['deficiency', 'PA', 'noun'],
        'न्यूनता': ['deficiency', 'SKRT', 'noun'],
        
        #'जमहूरियत': ['democracy', 'PA', 'noun'],          # UNDERREPRESENTED WORD IN THE CONTROL CORPUS
        #'लोकतंत्र': ['democracy', 'SKRT', 'noun'],         
        
        'क़िस्मत': ['destiny', 'PA', 'noun'],
        'भाग्य': ['destiny', 'SKRT', 'noun'],
        
        'मुख़्तलिफ़': ['different', 'PA', 'adj'],
        'विभिन्न': ['different', 'SKRT', 'adj'],
        
        'मुश्किल': ['difficult', 'PA', 'adj'],
        'कठिन': ['difficult', 'SKRT', 'adj'],
        
        'दरवाज़ा': ['door', 'PA', 'noun'],
        'द्वार': ['door', 'SKRT', 'noun'],
        
        'ख़्वाब': ['dream, vision', 'PA', 'noun'],
        'सपना': ['dream, vision', 'SKRT', 'noun'],
        
        'आसान': ['easy', 'PA', 'adj'],               
        'सरल': ['easy', 'SKRT', 'adj'],              
        
        'ज़मीन': ['earth, soil', 'PA', 'noun'],         
        'धरती': ['earth, soil', 'SKRT', 'noun'],
        
        'असर': ['effect, influence', 'PA', 'noun'],            
        'प्रभाव': ['effect, influence', 'SKRT', 'noun'],
        
        'दुश्मन': ['enemy', 'PA', 'noun'],            
        'शत्रु': ['enemy', 'SKRT', 'noun'],
           
        'इम्तिहान': ['examination', 'PA', 'noun'],            
        'परीक्षा': ['examination', 'SKRT', 'noun'],             
 
        'ख़ानदान': ['family', 'PA', 'noun'],            
        'परिवार': ['family', 'SKRT', 'noun'], 
        
        'मशहूर': ['famous', 'PA', 'adj'],             
        'प्रसिद्ध': ['famous', 'SKRT', 'adj'],  
        
        'इनायत': ['favour, kindness', 'PA', 'noun'],             
        'उपकार': ['favour, kindness', 'SKRT', 'noun'], 
        
        'दहशत': ['fear', 'PA', 'noun'],             
        'आतंक': ['fear', 'SKRT', 'noun'], 
        
        'त्योहार': ['festival', 'PA', 'adj'],             
        'उत्सव': ['festival', 'SKRT', 'adj'],
        
        'ख़त्म': ['finished', 'PA', 'adj'],             
        'समाप्त': ['finished', 'SKRT', 'adj'],   
        
        'सैलाब': ['flood, inondation', 'PA', 'noun'],            
        'प्लावन': ['flood, inondation', 'SKRT', 'noun'],
        
        'औपचारिक': ['formal/official', 'SKRT', 'adj'],            
        'अधिकारिक': ['formal/official', 'SKRT', 'adj'],        
         
        'आज़ादी': ['freedom', 'PA', 'noun'],           
        'स्वतंत्रता': ['freedom', 'SKRT', 'noun'],      
            
        'दोस्त': ['friend', 'PA', 'noun'],              
        'मित्र': ['friend', 'SKRT', 'noun'],            
        
        'नुक़सान': ['harm, damage', 'PA', 'noun'],
        'हानि': ['harm, damage', 'SKRT', 'noun'],
        
        'सेहत': ['health', 'PA', 'noun'],       
        'स्वस्थता': ['health', 'SKRT', 'noun'],  
        
        'दिल': ['heart', 'PA', 'noun'],              
        'हृदय': ['heart', 'SKRT', 'noun'],  
        
        'मदद': ['help, aid', 'PA', 'noun'],              
        'सहायता': ['help, aid', 'SKRT', 'noun'], 
        
        'इज़्ज़त': ['honour, esteem', 'PA', 'noun'],            
        'सम्मान': ['honour, esteem', 'SKRT', 'noun'], 
        
        'उम्मीद': ['hope, expectation', 'PA', 'noun'],              
        'आशा': ['hope, expectation', 'SKRT', 'noun'], 
        
        'ख़याल': ['idea', 'PA', 'noun'],              
        'विचार': ['idea', 'SKRT', 'noun'],    
        
        'बीमार': ['ill', 'PA', 'adj'],        
        #'मरीज़': ['ill', 'PA', 'adj'],                   # PODAĆ JAKO PRZYKŁAD BIMAR / MARIZ jako ILL. ALE BIMAR CZESTSZE.
        'रोगी': ['ill', 'SKRT', 'adj'],
        
        'बीमारी': ['illness', 'PA', 'noun'],        
        'रोग': ['illness', 'SKRT', 'noun'],
        #'व्याधि': ['illness', 'SKRT', 'noun'],
        
        'तसव्वुर': ['imagination', 'PA', 'adj'],        
        'कल्पना': ['imagination', 'SKRT', 'adj'],
        
        'अहम': ['important', 'PA', 'adj'],        
        'महत्वपूर्ण': ['important', 'SKRT', 'adj'],
        
        'इत्तला': ['information', 'PA', 'adj'],        
        'सूचना': ['information', 'SKRT', 'adj'],
        
        'जज़ीरा': ['island', 'PA', 'noun'],           
        'द्वीप': ['island', 'SKRT', 'noun'], 
         
        'ज़ेवर': ['jewelry (piece of)', 'PA', 'noun'],       
        'आभूषण': ['jewelry (piece of)', 'SKRT', 'noun'], 
            
        'ज़बान': ['language', 'PA', 'noun'],
        'भाषा': ['language', 'SKRT', 'noun'],
        
        'ख़त': ['letter', 'PA', 'noun'],
        'चिठ्ठी': ['letter', 'SKRT', 'noun'],
        
        'ज़िंदगी': ['life', 'PA', 'noun'],
        'जीवन': ['life', 'SKRT', 'noun'],
        
        'इश्क़': ['love', 'PA', 'noun'],
        'प्रेम': ['love', 'SKRT', 'noun'],
                
        'इंसान': ['human being', 'PA', 'noun'],
        'मनुष्य': ['human being', 'SKRT', 'noun'],
        
        'गोश्त': ['meat', 'PA', 'noun'],
        'मांस': ['meat', 'SKRT', 'noun'],
        
        'दवा': ['medicine', 'PA', 'noun'],
        'औषधि': ['medicine', 'SKRT', 'noun'],
        
        'मुलाक़ात': ['meeting', 'PA', 'noun'],
        #'भेंट': ['meeting', 'SKRT', 'noun'],   # BHENT TO NIE SANSKRYT?
        'मिलाप': ['meeting', 'SKRT', 'noun'],
        
        'महज़': ['merely', 'PA', 'adv'],
        'मात्र': ['merely', 'SKRT', 'adv'],
        
        'आईना': ['mirror', 'PA', 'noun'],
        'दर्पण': ['mirror', 'SKRT', 'noun'],
        
        'महीना': ['month', 'PA', 'noun'],
        'मास': ['month', 'SKRT', 'noun'],
        
        'ज़्यादा': ['more, many', 'PA', 'adj/adv'],
        'अधिक': ['more, many', 'SKRT', 'adj/adv'],
        
        'क़ौम' : ['nation', 'PA', 'noun'],
        'राष्ट्र': ['nation', 'SKRT', 'noun'],
        
        'सिर्फ़' : ['only', 'PA', 'adv'],
        'केवल': ['only', 'SKRT', 'adv'],
        
        'मौक़ा': ['opportunity, occasion', 'PA', 'noun'],              
        'अवसर': ['opportunity, occasion', 'SKRT', 'noun'],
        
        'हुक्म': ['order, command', 'PA', 'noun'],                   
        'आदेश': ['order, command', 'SKRT', 'noun'],
        
        'हिस्सा': ['part, section, division', 'PA', 'noun'],                  
        'भाग': ['part, section, division', 'SKRT', 'noun'],
        
        'तक़सीम' : ['partition', 'PA', 'noun'],                  
        'विभाजन': ['partition', 'SKRT', 'noun'],
        
        'सब्र': ['patience', 'PA', 'noun'],                  
        'धैर्य': ['patience', 'SKRT', 'noun'],
        #'क्षांति': ['patience', 'SKRT', 'noun'],        #### CIEKAWY PRZYKŁAD सहन
        
        'फ़ीसदी': ['per cent, percentage', 'PA', 'adv'],    
        'प्रतिशत': ['per cent, percentage', 'SKRT', 'adv'],
        
        'इजाज़त': ['permision', 'PA', 'noun'],                
        'अनुमति': ['permision', 'SKRT', 'noun'],
        
        'शख़्स': ['person', 'PA', 'noun'],                   
        'व्यक्ति': ['person', 'SKRT', 'noun'],
        
        'जगह': ['place', 'PA', 'noun'],                    
        'स्थान': ['place', 'SKRT', 'noun'],
        
        'साज़िश': ['plot, conspiracy', 'PA', 'noun'],    
        'षड्यंत्र': ['plot, conspiracy', 'SKRT', 'noun'],
        
        'मुमकिन': ['possible', 'PA', 'adj'],                    
        'संभव': ['possible', 'SKRT', 'adj'],            
        
        'ज़हर' : ['poison', 'PA', 'noun'],
        'विष': ['poison', 'SKRT', 'noun'],
        
        'मसला': ['problem', 'PA', 'noun'],
        'समस्या': ['problem', 'SKRT', 'noun'],
        
        'सवाल': ['question', 'PA', 'adj'],
        'प्रश्न': ['question', 'SKRT', 'noun'],
        
        'असली': ['real', 'PA', 'adj'],
        #'वास्तव': ['real', 'SKRT', 'adj'],
        'वास्तविक': ['real', 'SKRT', 'adj'],
        
        'हक़ीक़त': ['reality', 'PA', 'noun'],
        'वास्तविकता': ['reality', 'PA', 'noun'],
        
        'मज़हब': ['religion', 'PA', 'noun'],
        'धर्म': ['religion', 'SKRT', 'noun'],
        
        'गुज़ारिश': ['request', 'PA', 'noun'],
        'निवेदन': ['request', 'SKRT', 'noun'],
        
        'आराम': ['rest', 'PA', 'noun'],
        'विश्राम': ['rest', 'SKRT', 'noun'],
        
        'नतीजा': ['result', 'PA', 'noun'],
        'परिणाम': ['result', 'SKRT', 'noun'],     
        
        'दरिया': ['river', 'PA', 'noun'],
        'नदी': ['river', 'SKRT', 'noun'],
        
        'कमरा': ['room', 'PA', 'noun'],
        'कक्ष': ['room', 'SKRT', 'noun'],
        
        #'राज़ी': ['satisfied', 'PA', 'adj'],         
        #'तुष्टी': ['satisfied', 'SKRT', 'adj'],       # UNDERREPRESENTED WORD IN THE CONTROL CORPUS
        
        'वज़ीफ़ा': ['scholarship', 'PA', 'noun'],
        'छात्रवृत्ति': ['scholarship', 'SKRT', 'noun'],
        
        'नौकर': ['servant', 'PA', 'noun'],
        'सेवक': ['servant', 'SKRT', 'noun'],
        
        'शर्म': ['shame', 'PA', 'noun'],
        'लज्जा': ['shame', 'SKRT', 'noun'],
        
        'ख़ामोश': ['silent', 'PA', 'adj'],
        'मौन': ['silent', 'SKRT', 'adj'],
        
        'फ़लक': ['sky, heaven', 'PA', 'noun'],
        'आकाश': ['sky, heaven', 'SKRT', 'noun'],
        
        'बेवकूफ़': ['stupid', 'PA', 'adj'],
        'मूर्ख': ['stupid', 'SKRT', 'adj'],
        
        'अजीब': ['strange', 'PA', 'adj'],
        'अनोखा': ['strange', 'SKRT', 'adj'],
        
        'ज़िद': ['stubborness, obstinacy', 'PA', 'noun'],
        'हठ': ['stubborness, obstinacy', 'SKRT', 'noun'],
        
        'सिपाही': ['soldier', 'PA', 'noun'],
        'सैनिक': ['soldier', 'SKRT', 'noun'],
        
        'ग़म': ['sorrow', 'PA', 'noun'],
        'दुख': ['sorrow', 'SKRT', 'noun'],
        
        'ताक़त': ['strength, power', 'PA', 'noun'],
        'शक्ति': ['strength, power', 'SKRT', 'noun'],
        
        'बारीक': ['subtle, delicate', 'PA', 'noun'],        
        'सूक्ष्म': ['subtle, delicate', 'SKRT', 'noun'],
        
        'ख़ुदकुशी': ['suicide', 'PA', 'noun'],
        'आत्महत्या': ['suicide', 'SKRT', 'noun'],
        
        'आफ़ताब': ['sun', 'PA', 'noun'],
        'सूर्य': ['sun', 'SKRT', 'noun'],
        
        'सफ़र' : ['travel', 'PA', 'noun'],
        'यात्रा' : ['travel', 'SKRT', 'noun'],
        
        'मुसाफ़िर' : ['traveller', 'PA', 'noun'],
        'सवारी': ['traveller', 'SKRT', 'noun'],
        
        'दरख़्त': ['tree', 'PA', 'noun'],
        'वृक्ष': ['tree', 'SKRT', 'noun'],
        
        'तकलीफ़': ['trouble, difficulty', 'PA', 'noun'],
        'कष्ट': ['trouble, difficulty', 'SKRT', 'noun'],
        
        'वक़्त': ['time', 'PA', 'noun'],
        'समय': ['time', 'SKRT', 'noun'],
        
        'बेहोश': ['unconscious', 'PA', 'adj'],
        'अचेत': ['unconscious', 'SKRT', 'adj'],
        
        'इस्तेमाल': ['use', 'PA', 'noun/verb'],
        'प्रयोग': ['use', 'SKRT', 'noun/verb'],
        
        'ख़ाली': ['void', 'PA', 'noun'],
        'रिक्त': ['void', 'SKRT', 'noun'],
        
        'इंतज़ार': ['wait', 'PA', 'noun/verb'],
        'प्रतीक्षा': ['wait', 'SKRT', 'noun/verb'],
        
        'जंग': ['war', 'PA', 'noun'],
        'युद्ध': ['war', 'SKRT', 'noun'],
        
        'आब': ['water', 'PA', 'noun'],
        'पानी': ['water', 'SKRT', 'noun'],
        
        'हफ़्ता': ['week', 'PA', 'noun'],
        'सप्ताह': ['week', 'SKRT', 'noun'],
        
        'ख़ैरियत': ['well-being', 'PA', 'noun'],
        'क्षेम': ['well-being', 'SKRT', 'noun'],
        
        'गवाह': ['witness', 'PA', 'noun'],
        'साक्षी': ['witness', 'SKRT', 'noun'],
        
        'औरत': ['woman', 'PA', 'noun'],
        'महिला': ['woman', 'SKRT', 'noun'],
        
        'लफ़्ज़': ['word', 'PA', 'noun'],
        'शब्द': ['word', 'SKRT', 'noun'],
        
        'साल': ['year', 'PA', 'noun'],
        'वर्ष': ['year', 'SKRT', 'noun']
        }
    
    def __init__(self):
        """Initialize the RawVocabulary class."""
        pass
    
    def display_stats(self):
        """Display comprehensive statistics about the vocabulary dataset."""
        
        raw_vocabulary_length = len(self.synonym_list)
        unique_meanings = len(set([ self.synonym_list[elem][0] for elem in self.synonym_list ]))
        SKRT_words = len([ self.synonym_list[elem][1] for elem in self.synonym_list if self.synonym_list[elem][1]=='SKRT' ])
        PA_words = len([ self.synonym_list[elem][1] for elem in self.synonym_list if self.synonym_list[elem][1]=='PA' ])
        nouns = len([ self.synonym_list[elem][2] for elem in self.synonym_list if self.synonym_list[elem][2]=='noun' ])
        adjectives = len([ self.synonym_list[elem][2] for elem in self.synonym_list if self.synonym_list[elem][2]== 'adj' ])
        ppos = len([ self.synonym_list[elem][2] for elem in self.synonym_list if self.synonym_list[elem][2]== 'ppos' ])
        others = len([ self.synonym_list[elem][2] for elem in self.synonym_list if self.synonym_list[elem][2] not in ['noun', 'adj', 'ppos'] ])
        
        # Colored headers with same color scheme as load_model
        headers = [
            f'{CYAN}Raw vocabulary length{RESET}',
            f'{YELLOW}Number of Unique Meanings{RESET}',
            f'{SKRT_COLOR}Sanskrit Words{RESET}',
            f'{PA_COLOR}Perso-Arabic Words{RESET}',
            f'{MAGENTA}SKRT + PA Words{RESET}',
            f'{WHITE}Nouns{RESET}',
            f'{WHITE}Adj.{RESET}',
            f'{WHITE}Ppos{RESET}',
            f'{WHITE}Other POS{RESET}'
        ]
        
        # Colored values matching headers
        table = [[
            f"{CYAN}{raw_vocabulary_length}{RESET}",
            f"{YELLOW}{unique_meanings}{RESET}",
            f"{SKRT_COLOR}{SKRT_words}{RESET}",
            f"{PA_COLOR}{PA_words}{RESET}",
            f"{MAGENTA}{SKRT_words+PA_words}{RESET}",
            nouns,
            adjectives,
            ppos,
            others
        ]]
        
        tab = PrettyTable()
        tab.title = f'{BLUE}Raw Vocabulary Specifications{RESET}'
        tab.field_names = headers
        tab.add_rows(table)
        print(tab)


    def convert_to_df(self):
        """
        Convert the synonym list to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with columns Word, Meaning, Origin, POS
        """
        vocab_dict = {
            'Word': list(self.synonym_list.keys()),
            'Meaning': [self.synonym_list[word][0] for word in self.synonym_list],
            'Origin': [self.synonym_list[word][1] for word in self.synonym_list],
            'POS': [self.synonym_list[word][2] for word in self.synonym_list],
        }
        vocabulary = pd.DataFrame(vocab_dict)
        return vocabulary
    
    def build_embeddings(self, raw_vocabulary, model_loaded):
        """
        Build word embeddings from the vocabulary using a pre-trained Word2Vec model.
        
        Args:
            raw_vocabulary: DataFrame containing the vocabulary
            model_loaded: Pre-trained Word2Vec model
            
        Returns:
            Tuple containing cleaned vocabulary DataFrame, embedding columns, and mapping dictionary
        """
        
        model_loaded = model_loaded
        errors = []
        embeddings = []
        frequency = []

        for word in list(raw_vocabulary['Word']):
            try:
                embeddings += [model_loaded.wv[word]]
            except:
                embeddings += [nan]
                freq = 0
                errors += [[word]]
            else:
                freq = model_loaded.wv.get_vecattr(word, "count")
                
            frequency += [freq]
        
        vocabulary = raw_vocabulary.copy()

        vocabulary['Freq'] = frequency
        vocabulary["Embedding"] = embeddings
        
        # Drop the empty records i.e. the words without word embeddings
        vocabulary_clean = vocabulary.dropna()
        vocabulary_clean.head()

        if len(raw_vocabulary) == len(embeddings):
            overall_result = f'{GREEN}OK{RESET}'
        else:
            overall_result = f'{RED}KO{RESET}'

        embeddings_summary = PrettyTable()
        embeddings_summary.title = f'{BLUE}Embedding Computation Summary{RESET}'
        
        # Colored headers
        headers = [
            f'{CYAN}Raw Vocabulary{RESET}',
            f'{YELLOW}Number of embeddings{RESET}',
            f'{RED}No embeddings found for{RESET}',
            f'{MAGENTA}After dropna{RESET}',
            f'{WHITE}Overall result{RESET}'
        ]
        
        # Colored values
        table = [[
            f"{CYAN}{len(vocabulary)} words{RESET}",
            f"{YELLOW}{len(embeddings)} words{RESET}",
            f"{RED}{len(errors)} words{RESET}",
            f"{MAGENTA}{len(vocabulary_clean)}{RESET}",
            overall_result
        ]]
        
        embeddings_summary.field_names = headers
        embeddings_summary.add_rows(table)
        print(embeddings_summary)
        
        # Display errors if there are some:
        if errors:
            errors_summary = PrettyTable()
            errors_summary.title = f'{RED}Missing embeddings — Words not recognized: {len(errors)} word(s){RESET}'
            headers = [f'{RED}Words{RESET}']
            errors_summary.field_names = headers
            
            # Color the error words based on their origin
            colored_errors = []
            for error_word in errors:
                word = error_word[0]
                # Find origin of the error word
                origin_row = raw_vocabulary[raw_vocabulary['Word'] == word]
                if not origin_row.empty:
                    origin = origin_row['Origin'].iloc[0]
                    if origin == 'PA':
                        colored_word = f"{PA_COLOR}{word}{RESET}"
                    else:  # SKRT
                        colored_word = f"{SKRT_COLOR}{word}{RESET}"
                    colored_errors.append([colored_word])
                else:
                    colored_errors.append([word])  # No color if origin not found
            
            errors_summary.add_rows(colored_errors)
            print(errors_summary)
            
        # Convert word embedding vector into list of integers
        # The embeddings columns is the size of the vector of our model
        vector_size = model_loaded.vector_size
        embedding_cols = list(range(0,vector_size))                     # We will need the list of the embedding columns to train our model
        embedding_cols = list(map(str, embedding_cols))
        vocabulary_clean[embedding_cols] = vocabulary_clean['Embedding'].tolist()
        vocabulary_clean.drop('Embedding', axis=1, inplace=True)
        
        # Adding Origin_code to the dataframe:
        vocabulary_clean.insert(loc = 3,
                            column = 'Origin_code',
                            value = vocabulary_clean["Origin"].astype('category'))
        vocabulary_clean['Origin_code'] = vocabulary_clean["Origin"].astype('category').cat.codes
        
        mapping_dict = {}
        for elem in vocabulary_clean['Origin_code'].unique():
            mapping_dict[elem] = vocabulary_clean[vocabulary_clean['Origin_code'] == elem]['Origin'].unique()[0]
        return vocabulary_clean, embedding_cols, mapping_dict


    def give_synonyms(self, word, dataframe):
        """
        Find all synonyms for a given word.
        
        Args:
            word: The word to find synonyms for
            dataframe: DataFrame containing the vocabulary
            
        Returns:
            List of synonyms including the original word
        """
        meaning = dataframe[dataframe['Word'] == word]['Meaning'].iloc[0]
        temp = [word]
        temp += dataframe[dataframe['Meaning'] == meaning]['Word'].tolist()
        temp = list(dict.fromkeys(temp))
        return temp

    def give_similarity_freq_diff(self, synonyms, model_loaded):
        """
        Calculate similarity and frequency difference between two synonyms.
        
        Args:
            synonyms: List of two synonym words
            model_loaded: Pre-trained Word2Vec model
            
        Returns:
            Tuple of (similarity_score, frequency_difference)
        """
        similarity = model_loaded.wv.similarity(synonyms[0], synonyms[1])
        freq_0 = model_loaded.wv.get_vecattr(synonyms[0], "count")
        freq_1 = model_loaded.wv.get_vecattr(synonyms[1], "count")
        freq_diff = freq_0 - freq_1
        return similarity, freq_diff


    def add_similarity_freq_diff(self, dataframe, model_loaded):
        """
        Add Word2Vec similarity coefficients and frequency differences to the dataframe.
        
        This method computes similarity scores between synonym pairs and their frequency
        differences, adding them as new columns to the input dataframe.
        
        Args:
            dataframe: DataFrame containing the vocabulary with embeddings
            model_loaded: Pre-trained Word2Vec model
            
        Returns:
            DataFrame with added Similarity and Freq_diff columns
        """
        similarity = []
        freq_diff = []
        no_similarity = 0
        words_without_synonyms = []
        words_with_more_synonyms = []
        words = list(dataframe['Word'])
        for word in words:
            synonyms = self.give_synonyms(word, dataframe)
            if len(synonyms) == 2:
                word_similarity, word_freq_diff = self.give_similarity_freq_diff(synonyms, model_loaded)
            elif len(synonyms) == 1:
                #print(f'No synonym found for: {synonyms} - similarity will be set to 0')
                word_similarity = 0
                word_freq_diff = 0
                no_similarity += 1
                words_without_synonyms += synonyms
            elif len(synonyms) > 2:
                #print(f'More than 2 words for one meaning: {synonyms} — similarity will be set to 0')
                word_similarity = 0
                word_freq_diff = 0
                no_similarity += 1
                if synonyms not in words_with_more_synonyms:
                    words_with_more_synonyms += [synonyms]
            similarity += [word_similarity]
            freq_diff += [word_freq_diff]
                
        dataframe.insert(loc = 5,
                            column = 'Similarity',
                            value = similarity)
        dataframe.insert(loc = 7,
                            column = 'Freq_diff',
                            value = freq_diff)
        similarity = pd.DataFrame(similarity)
        errors_summary = PrettyTable()
        
        # Conditional coloring based on error count
        if no_similarity == 0:
            # Green table when no errors
            errors_summary.title = f'{GREEN}Summary for similarity computation: {no_similarity} errors{RESET}'
            header = [f'{GREEN}Word{RESET}', f'{GREEN}Error{RESET}']
        else:
            # Red table when errors exist
            errors_summary.title = f'{RED}Summary for similarity computation: {no_similarity} errors{RESET}'
            header = [f'{RED}Word{RESET}', f'{RED}Error{RESET}']
        
        errors_summary.field_names = header
        rows = []
        
        # Color rows based on whether there are errors
        if no_similarity == 0:
            # Green rows when no errors (but this case shouldn't have any rows)
            for word in words_without_synonyms:
                rows += [[f'{GREEN}{word}{RESET}', f'{GREEN}No synonym found{RESET}']]
            for words in words_with_more_synonyms:
                text = ', '.join(words)
                rows += [[f'{GREEN}{text}{RESET}', f'{GREEN}More than 2 words for one meaning{RESET}']]
        else:
            # Red rows when errors exist
            for word in words_without_synonyms:
                rows += [[f'{RED}{word}{RESET}', f'{RED}No synonym found{RESET}']]
            for words in words_with_more_synonyms:
                text = ', '.join(words)
                rows += [[f'{RED}{text}{RESET}', f'{RED}More than 2 words for one meaning{RESET}']]
        
        errors_summary.add_rows(rows)
        print(errors_summary)
        return dataframe