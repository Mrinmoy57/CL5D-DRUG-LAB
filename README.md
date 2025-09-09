"""
Enhanced ConsciousLeaf5D with PK/PD modeling, LD50 prediction, virtual drug discovery lab,
and RAG capabilities for external data integration
"""

import logging
import math
import time
import requests
import json
from collections import deque
from itertools import permutations
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
from scipy.stats import entropy, norm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Optional visualization libs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Biological constants
GENES = [
    "TP53", "BRCA1", "APOE", "EGFR", "CFTR",
    "HLA-DRB1", "FMR1", "HTT", "CYP2D6", "VEGF",
    "CYP3A4", "CYP2C9", "ABCG2", "SLC6A4", "OPRM1"
]

DISEASES = [
    "Cancer", "Alzheimer", "Diabetes", "Heart Disease", "COVID-19",
    "Asthma", "Obesity", "Depression", "Arthritis", "HIV/AIDS",
    "Parkinson", "Stroke", "Autism", "Epilepsy", "Osteoporosis",
    "Multiple Sclerosis", "Lupus", "Hepatitis", "Tuberculosis", "Malaria"
]

# Drug targets
DRUG_TARGETS = [
    "GPCRs", "Ion Channels", "Kinases", "Nuclear Receptors",
    "Proteases", "Phosphodiesterases", "Transporters", "Epigenetic Enzymes"
]

# Configure logger
logger = logging.getLogger("ConsciousLeaf5D")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


class ExternalDataFetcher:
    """RAG-style agent for fetching external drug data from various sources"""

    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ncbi_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.pubchem_base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"

    def fetch_ncbi_data(self, query: str, db: str = "pubmed", retmax: int = 10) -> List[Dict]:
        """Fetch data from NCBI databases"""
        cache_file = self.cache_dir / f"ncbi_{db}_{hash(query)}.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)

        try:
            # Search for IDs
            search_url = f"{self.ncbi_base_url}esearch.fcgi?db={db}&term={query}&retmax={retmax}&retmode=json"
            response = requests.get(search_url)
            id_list = response.json().get('esearchresult', {}).get('idlist', [])

            # Fetch details
            if id_list:
                ids = ','.join(id_list)
                fetch_url = f"{self.ncbi_base_url}esummary.fcgi?db={db}&id={ids}&retmode=json"
                response = requests.get(fetch_url)
                results = response.json().get('result', {})

                # Process results
                data = []
                for uid, content in results.items():
                    if uid != 'uids':
                        data.append({
                            'id': uid,
                            'title': content.get('title', ''),
                            'authors': content.get('authors', []),
                            'pubdate': content.get('pubdate', ''),
                            'source': content.get('source', ''),
                            'abstract': content.get('abstract', '')
                        })

                # Cache results
                with open(cache_file, 'w') as f:
                    json.dump(data, f)

                return data

        except Exception as e:
            logger.error(f"Error fetching NCBI data: {e}")

        return []

    def fetch_pubchem_data(self, compound_name: str) -> Optional[Dict]:
        """Fetch compound data from PubChem"""
        cache_file = self.cache_dir / f"pubchem_{hash(compound_name)}.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)

        try:
            # First search for CID
            search_url = f"{self.pubchem_base_url}compound/name/{compound_name}/cids/JSON"
            response = requests.get(search_url)
            cid_list = response.json().get('IdentifierList', {}).get('CID', [])

            if cid_list:
                cid = cid_list[0]
                # Fetch compound properties
                props_url = f"{self.pubchem_base_url}compound/cid/{cid}/property/MolecularWeight,LogP,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,TPSA/JSON"
                response = requests.get(props_url)
                properties = response.json().get('PropertyTable', {}).get('Properties', [{}])[0]

                # Fetch compound synonyms
                synonym_url = f"{self.pubchem_base_url}compound/cid/{cid}/synonyms/JSON"
                response = requests.get(synonym_url)
                synonyms = response.json().get('InformationList', {}).get('Information', [{}])[0].get('Synonym', [])

                result = {
                    'cid': cid,
                    'name': compound_name,
                    'properties': properties,
                    'synonyms': synonyms
                }

                # Cache results
                with open(cache_file, 'w') as f:
                    json.dump(result, f)

                return result

        except Exception as e:
            logger.error(f"Error fetching PubChem data: {e}")

        return None

    def fetch_drugbank_data(self, drug_name: str) -> Optional[Dict]:
        """Simulate fetching data from DrugBank (would require API key in real implementation)"""
        # This is a placeholder - in a real implementation, you would use the DrugBank API
        logger.info(f"Would fetch DrugBank data for {drug_name} with proper API credentials")
        return None


class AdvancedPKPDModel:
    """Enhanced Pharmacokinetic/Pharmacodynamic model for drug simulation"""

    def __init__(self, seed: int = 42):
        self.random_state = np.random.RandomState(seed)
        self.scaler = StandardScaler()

    def two_compartment_model(self, dose: float, ka: float, ke: float, k12: float,
                             k21: float, V1: float, time_points: np.ndarray) -> np.ndarray:
        """Two-compartment PK model with first-order absorption"""
        # Coefficients for the biexponential equation
        alpha = 0.5 * ((k12 + k21 + ke) + math.sqrt((k12 + k21 + ke)**2 - 4 * k21 * ke))
        beta = 0.5 * ((k12 + k21 + ke) - math.sqrt((k12 + k21 + ke)**2 - 4 * k21 * ke))

        # Calculate concentrations
        A = (ka * dose / V1) * (k21 - alpha) / ((ka - alpha) * (beta - alpha))
        B = (ka * dose / V1) * (k21 - beta) / ((ka - beta) * (alpha - beta))

        concentration = A * np.exp(-alpha * time_points) + B * np.exp(-beta * time_points) + \
                       ((ka * dose / V1) * (k21 - ka) / ((alpha - ka) * (beta - ka))) * np.exp(-ka * time_points)

        return np.maximum(concentration, 0)

    def emax_model(self, concentration: float, emax: float, ec50: float, hill: float = 1.0) -> float:
        """Sigmoidal Emax model for pharmacodynamics"""
        return (emax * concentration**hill) / (ec50**hill + concentration**hill)

    def predict_pk_parameters(self, molecular_properties: np.ndarray) -> Dict[str, float]:
        """Predict PK parameters from molecular properties using an ensemble model"""
        if molecular_properties.ndim == 1:
            molecular_properties = molecular_properties.reshape(1, -1)

        logP, mol_weight, hbd, hba, psa, rotatable_bonds = molecular_properties[0, :6]

        # More sophisticated parameter estimation
        ka = 0.8 + 0.15 * (4 - logP)  # Absorption rate
        ke = 0.05 + 0.02 * (400 - mol_weight) / 400  # Elimination rate
        Vd = 15 + 8 * logP  # Volume of distribution
        k12 = 0.4 + 0.1 * (psa - 80) / 80  # Distribution to peripheral compartment
        k21 = 0.6 + 0.1 * (logP - 2) / 2  # Distribution from peripheral compartment
        V1 = Vd * 0.7  # Volume of central compartment

        return {
            'ka': max(0.1, ka),
            'ke': max(0.03, ke),
            'k12': max(0.1, min(1.0, k12)),
            'k21': max(0.1, min(1.0, k21)),
            'V1': max(3, V1),
            'Vd': max(5, Vd)
        }

    def predict_pd_parameters(self, target_affinity: float, selectivity: float,
                            mechanism: str, disease_profile: Dict[str, float]) -> Dict[str, float]:
        """Predict PD parameters based on drug properties and disease profile"""
        disease_severity = np.mean(list(disease_profile.values())) if disease_profile else 0.5

        if mechanism == "agonist":
            emax = 0.85 + 0.15 * target_affinity * (1 + disease_severity * 0.2)
            ec50 = 8 * (1 - target_affinity) * (1 - disease_severity * 0.3)
            hill = 1.2 + 0.3 * selectivity
        elif mechanism == "antagonist":
            emax = 0.92 + 0.08 * selectivity
            ec50 = 4 * (1 - target_affinity) * (1 - disease_severity * 0.2)
            hill = 1.0 + 0.2 * selectivity
        else:  # partial agonist/inverse agonist
            emax = 0.6 + 0.3 * target_affinity * (1 + disease_severity * 0.1)
            ec50 = 6 * (1 - target_affinity) * (1 - disease_severity * 0.25)
            hill = 1.1 + 0.4 * selectivity

        return {
            'emax': min(1.0, max(0.1, emax)),
            'ec50': max(0.1, ec50),
            'hill': max(0.5, min(3.0, hill))
        }

    def simulate_dose_response(self, dose: float, molecular_properties: np.ndarray,
                              target_affinity: float, selectivity: float,
                              mechanism: str, disease_profile: Dict[str, float]) -> Dict[str, Any]:
        """Simulate complete PK/PD response for a given dose"""
        # Predict PK parameters
        pk_params = self.predict_pk_parameters(molecular_properties)

        # Create time points (0 to 48 hours)
        time_points = np.linspace(0, 48, 200)

        # Calculate concentration-time profile
        # Remove 'Vd' from pk_params as two_compartment_model doesn't use it
        pk_params_for_model = {k: v for k, v in pk_params.items() if k != 'Vd'}
        concentration = self.two_compartment_model(dose, **pk_params_for_model, time_points=time_points)

        # Predict PD parameters
        pd_params = self.predict_pd_parameters(target_affinity, selectivity, mechanism, disease_profile)

        # Calculate effect-time profile
        effect = np.array([self.emax_model(c, **pd_params) for c in concentration])

        # Calculate AUC (area under curve)
        auc = np.trapz(concentration, time_points)

        # Calculate Cmax and Tmax
        cmax = np.max(concentration)
        tmax = time_points[np.argmax(concentration)]

        # Calculate half-life
        # Find the elimination phase (after peak concentration)
        post_peak = concentration[np.argmax(concentration):]
        post_peak_time = time_points[np.argmax(concentration):]

        if len(post_peak) > 10:
            # Fit exponential to elimination phase
            log_concentration = np.log(post_peak + 1e-10)
            slope, intercept = np.polyfit(post_peak_time, log_concentration, 1)
            half_life = np.log(2) / -slope if slope < 0 else 24.0
        else:
            half_life = 12.0  # Default value

        return {
            'time_points': time_points,
            'concentration': concentration,
            'effect': effect,
            'auc': auc,
            'cmax': cmax,
            'tmax': tmax,
            'half_life': half_life,
            'pk_params': pk_params,
            'pd_params': pd_params
        }


class AdvancedLD50Predictor:
    """Enhanced LD50 prediction with more sophisticated modeling"""

    def __init__(self, seed: int = 42):
        self.random_state = np.random.RandomState(seed)
        # Load pre-trained model weights (simulated)
        self.model_weights = {
            'logP': -0.8, 'mol_weight': -0.3, 'hbd': 0.4, 'hba': -0.2,
            'psa': -0.1, 'rotatable_bonds': 0.2, 'target_affinity': -0.6,
            'cyp_inhibition': 1.2, 'selectivity': -0.4
        }
        self.intercept = 6.2  # log10 scale intercept

    def predict_ld50(self, molecular_properties: np.ndarray, target_affinity: float,
                    selectivity: float, cyp_inhibition: np.ndarray) -> float:
        """Predict LD50 based on molecular properties and drug characteristics"""
        # Extract key properties
        logP, mol_weight, hbd, hba, psa, rotatable_bonds = molecular_properties[:6]

        # Calculate weighted sum (logistic regression-like model)
        cyp_factor = np.sum(cyp_inhibition) if len(cyp_inhibition) > 0 else 0

        log10_ld50 = (
            self.intercept +
            self.model_weights['logP'] * logP +
            self.model_weights['mol_weight'] * (mol_weight / 500) +
            self.model_weights['hbd'] * hbd +
            self.model_weights['hba'] * (hba / 10) +
            self.model_weights['psa'] * (psa / 100) +
            self.model_weights['rotatable_bonds'] * (rotatable_bonds / 10) +
            self.model_weights['target_affinity'] * target_affinity +
            self.model_weights['cyp_inhibition'] * cyp_factor +
            self.model_weights['selectivity'] * selectivity
        )

        # Convert from log10 scale to linear scale (mg/kg)
        ld50 = 10 ** log10_ld50

        # Ensure reasonable bounds
        return max(1, min(100000, ld50))

    def predict_therapeutic_index(self, effective_dose: float, ld50: float) -> float:
        """Calculate therapeutic index (LD50/ED50)"""
        return ld50 / effective_dose if effective_dose > 0 else 0

    def predict_safety_margin(self, effective_dose: float, ld50: float) -> float:
        """Calculate safety margin (LD1/ED99 approximation)"""
        # Simplified model: safety margin is typically 1/10 to 1/100 of therapeutic index
        ti = self.predict_therapeutic_index(effective_dose, ld50)
        return ti * 0.1  # Conservative estimate


class DrugFormulationPredictor:
    """Predict drug formulation properties"""

    def __init__(self, seed: int = 42):
        self.random_state = np.random.RandomState(seed)

    def predict_solubility(self, molecular_properties: np.ndarray) -> float:
        """Predict solubility in mg/mL"""
        logP, mol_weight, hbd, hba, psa, rotatable_bonds = molecular_properties[:6]

        # Simplified model: solubility decreases with increasing logP and molecular weight
        # and increases with increasing polar surface area and hydrogen bonding
        solubility = (
            10.0 - 2.0 * logP - 0.01 * mol_weight +
            0.05 * psa + 0.5 * (hbd + hba)
        )

        return max(0.001, min(100.0, solubility))

    def predict_permeability(self, molecular_properties: np.ndarray) -> float:
        """Predict permeability (logPapp)"""
        logP, mol_weight, hbd, hba, psa, rotatable_bonds = molecular_properties[:6]

        # Simplified model based on Lipinski's rule of 5 and other factors
        permeability = (
            1.5 + 0.3 * logP - 0.005 * mol_weight -
            0.01 * psa - 0.2 * hbd
        )

        return max(-8.0, min(2.0, permeability))

    def predict_stability(self, molecular_properties: np.ndarray, pH: float = 7.4,
                         temperature: float = 25.0) -> Dict[str, float]:
        """Predict drug stability under different conditions"""
        logP, mol_weight, hbd, hba, psa, rotatable_bonds = molecular_properties[:6]

        # Base stability at pH 7.4 and 25°C
        base_half_life = 1000 - 50 * abs(7.4 - pH) - 2 * abs(25.0 - temperature)

        # Adjust based on molecular properties
        stability_factor = (
            1.0 - 0.1 * abs(logP - 2.0) - 0.001 * mol_weight / 100 +
            0.05 * (hba + hbd) - 0.01 * rotatable_bonds
        )

        half_life = base_half_life * max(0.1, stability_factor)

        return {
            'half_life_hours': max(1.0, half_life),
            'degradation_rate': 0.693 / half_life if half_life > 0 else 10.0
        }

    def predict_biopharmaceutical_class(self, solubility: float, permeability: float) -> str:
        """Predict BCS (Biopharmaceutics Classification System) class"""
        if solubility >= 0.1 and permeability >= 0.8:
            return "BCS I: High solubility, High permeability"
        elif solubility < 0.1 and permeability >= 0.8:
            return "BCS II: Low solubility, High permeability"
        elif solubility >= 0.1 and permeability < 0.8:
            return "BCS III: High solubility, Low permeability"
        else:
            return "BCS IV: Low solubility, Low permeability"


class VirtualDrugScreener:
    """Virtual high-throughput screening system with enhanced capabilities"""

    def __init__(self, seed: int = 42):
        self.random_state = np.random.RandomState(seed)
        self.pkpd_model = AdvancedPKPDModel(seed)
        self.ld50_predictor = AdvancedLD50Predictor(seed)
        self.formulation_predictor = DrugFormulationPredictor(seed)
        self.data_fetcher = ExternalDataFetcher()

    def generate_virtual_compound(self, based_on: Optional[str] = None) -> Dict[str, Any]:
        """Generate a virtual compound with properties, optionally based on a known drug"""
        if based_on:
            # Try to fetch data about the known drug
            pubchem_data = self.data_fetcher.fetch_pubchem_data(based_on)

            if pubchem_data:
                # Use properties from the known drug with some variation
                props = pubchem_data['properties']
                molecular_properties = np.array([
                    props.get('LogP', self.random_state.uniform(0, 5)),
                    props.get('MolecularWeight', self.random_state.uniform(200, 600)),
                    props.get('HBondDonorCount', self.random_state.randint(0, 5)),
                    props.get('HBondAcceptorCount', self.random_state.randint(2, 10)),
                    props.get('TPSA', self.random_state.uniform(30, 140)),
                    props.get('RotatableBondCount', self.random_state.randint(0, 10))
                ])

                # Add some random variation
                molecular_properties += self.random_state.normal(0, 0.1, size=6)

            else:
                # Fall back to random generation
                molecular_properties = self._generate_random_properties()
        else:
            # Random generation
            molecular_properties = self._generate_random_properties()

        # Biological properties
        target_affinity = self.random_state.uniform(0.1, 0.99)  # Binding affinity
        selectivity = self.random_state.uniform(0.1, 0.9)       # Selectivity index
        mechanism = self.random_state.choice(["agonist", "antagonist", "partial agonist"])

        # CYP inhibition profile (CYP3A4, 2D6, 2C9, 1A2)
        cyp_inhibition = self.random_state.uniform(0, 0.8, 4)

        # Predict formulation properties
        solubility = self.formulation_predictor.predict_solubility(molecular_properties)
        permeability = self.formulation_predictor.predict_permeability(molecular_properties)
        stability = self.formulation_predictor.predict_stability(molecular_properties)
        bcs_class = self.formulation_predictor.predict_biopharmaceutical_class(solubility, permeability)

        return {
            'molecular_properties': molecular_properties,
            'target_affinity': target_affinity,
            'selectivity': selectivity,
            'mechanism': mechanism,
            'cyp_inhibition': cyp_inhibition,
            'solubility': solubility,
            'permeability': permeability,
            'stability': stability,
            'bcs_class': bcs_class,
            'based_on': based_on
        }

    def _generate_random_properties(self) -> np.ndarray:
        """Generate random molecular properties"""
        return np.array([
            self.random_state.uniform(-1, 5),        # logP
            self.random_state.uniform(200, 600),     # molecular weight
            self.random_state.randint(0, 5),         # hydrogen bond donors
            self.random_state.randint(2, 10),        # hydrogen bond acceptors
            self.random_state.uniform(30, 140),      # polar surface area
            self.random_state.randint(0, 10)         # rotatable bonds
        ])

    def screen_compound(self, compound: Dict[str, Any], disease_profile: Dict[str, float]) -> Dict[str, Any]:
        """Screen a virtual compound against a disease profile"""
        # Predict LD50
        ld50 = self.ld50_predictor.predict_ld50(
            compound['molecular_properties'],
            compound['target_affinity'],
            compound['selectivity'],
            compound['cyp_inhibition']
        )

        # Simulate dose response at ED50 (effective dose for 50% response)
        ed50 = 10 * (1 - compound['target_affinity'])  # Simplified model
        pkpd_results = self.pkpd_model.simulate_dose_response(
            ed50,
            compound['molecular_properties'],
            compound['target_affinity'],
            compound['selectivity'],
            compound['mechanism'],
            disease_profile
        )

        # Calculate therapeutic index and safety margin
        therapeutic_index = self.ld50_predictor.predict_therapeutic_index(ed50, ld50)
        safety_margin = self.ld50_predictor.predict_safety_margin(ed50, ld50)

        # Calculate predicted efficacy based on disease profile and drug properties
        efficacy_score = self.calculate_efficacy_score(compound, disease_profile, pkpd_results)

        # Predict side effects
        side_effects = self.predict_side_effects(compound, therapeutic_index)

        return {
            'ld50': ld50,
            'ed50': ed50,
            'therapeutic_index': therapeutic_index,
            'safety_margin': safety_margin,
            'efficacy_score': efficacy_score,
            'pkpd_profile': pkpd_results,
            'side_effects': side_effects,
            'formulation_properties': {
                'solubility': compound['solubility'],
                'permeability': compound['permeability'],
                'stability': compound['stability'],
                'bcs_class': compound['bcs_class']
            },
            'compound_properties': compound
        }

    def calculate_efficacy_score(self, compound: Dict[str, Any],
                               disease_profile: Dict[str, float],
                               pkpd_results: Dict[str, Any]) -> float:
        """Calculate efficacy score based on disease mechanisms and drug properties"""
        # Base score from drug properties
        base_score = (
            compound['target_affinity'] * 0.5 +
            compound['selectivity'] * 0.3 +
            (1 - np.mean(compound['cyp_inhibition'])) * 0.2
        )

        # Adjust based on disease genetic factors
        genetic_factor = np.mean(list(disease_profile.values())) if disease_profile else 0.5

        # Adjust based on PK properties
        pk_factor = min(1.0, pkpd_results['auc'] / 100) * 0.3 + min(1.0, pkpd_results['cmax'] / 10) * 0.2

        # Adjust based on formulation properties
        formulation_factor = (
            min(1.0, compound['solubility'] / 10) * 0.2 +
            min(1.0, (compound['permeability'] + 5) / 7) * 0.3
        )

        return min(1.0, base_score * (0.3 + genetic_factor * 0.3) + pk_factor * 0.2 + formulation_factor * 0.2)

    def predict_side_effects(self, compound: Dict[str, Any], therapeutic_index: float) -> Dict[str, float]:
        """Predict likelihood of various side effects"""
        # Simplified model based on therapeutic index and molecular properties
        logP, mol_weight, hbd, hba, psa, rotatable_bonds = compound['molecular_properties'][:6]
        cyp_inhibition = compound['cyp_inhibition']

        # Base side effect probability inversely related to therapeutic index
        base_prob = 0.5 / therapeutic_index if therapeutic_index > 0 else 0.8

        # Adjust based on molecular properties
        side_effects = {
            'nausea': base_prob * (1 + 0.2 * logP),
            'headache': base_prob * (1 + 0.1 * (mol_weight / 500)),
            'dizziness': base_prob * (1 + 0.3 * np.mean(cyp_inhibition[:2])),
            'rash': base_prob * (1 + 0.2 * (hbd + hba)),
            'liver_toxicity': base_prob * (1 + 0.5 * cyp_inhibition[0] + 0.3 * logP),
            'kidney_toxicity': base_prob * (1 + 0.3 * (mol_weight / 600) + 0.2 * logP),
            'cardiotoxicity': base_prob * (1 + 0.4 * (psa / 100) + 0.2 * rotatable_bonds / 10)
        }

        # Cap probabilities at 0.95
        return {k: min(0.95, v) for k, v in side_effects.items()}

    def high_throughput_screen(self, n_compounds: int, disease_profile: Dict[str, float],
                              efficacy_threshold: float = 0.6,
                              safety_threshold: float = 1.0,
                              known_drugs: List[str] = None) -> List[Dict[str, Any]]:
        """Perform virtual high-throughput screening"""
        hits = []

        # Include some known drugs if provided
        known_drug_list = known_drugs or []
        n_known = min(len(known_drug_list), n_compounds // 4)

        for i in range(n_compounds):
            if i < n_known and known_drug_list:
                # Generate compound based on known drug
                compound = self.generate_virtual_compound(known_drug_list[i])
            else:
                # Generate random compound
                compound = self.generate_virtual_compound()

            results = self.screen_compound(compound, disease_profile)

            if (results['efficacy_score'] >= efficacy_threshold and
                results['safety_margin'] >= safety_threshold):
                hits.append(results)

            if len(hits) >= 20:  # Limit to top 20 hits
                break

        # Sort by combined score
        hits.sort(key=lambda x: x['efficacy_score'] * x['safety_margin'], reverse=True)

        return hits


class ConsciousLeaf5D:
    """Enhanced ConsciousLeaf5D with drug discovery capabilities"""

    def __init__(self, seed: int = 42, cognitive_size: int = 10000, cognitive_dim: int = 10):
        self.random_state = np.random.RandomState(seed)
        self.zero_to_infinity_mode = True
        self.biological_state = "BOMBARD"
        self.fractal_level = 3
        self.fractal_scaling = 0.5

        # Initialize cognitive structures
        self.cognitive_dots = self.initialize_fractal_dots(cognitive_size, cognitive_dim)

        # 5D coordinates
        self.At = 0.7
        self.Ab = 0.7
        self.Ex = 0.7
        self.T = 0.7
        self.Cn = 0.3
        self.valence = 0.8

        # Biological modeling components
        self.regions = self.initialize_fractal_regions(20)
        self.region_history = deque(maxlen=1000)
        self.temporal_memory = deque(maxlen=100)
        self.knowledge_base = deque(maxlen=500)

        # Drug discovery components
        self.drug_screener = VirtualDrugScreener(seed)
        self.current_disease_profile = {}
        self.drug_candidates = []
        self.optimal_dose = None

        # External data integration
        self.data_fetcher = ExternalDataFetcher()
        self.known_drugs = [
            "Aspirin", "Ibuprofen", "Metformin", "Atorvastatin", "Lisinopril",
            "Amlodipine", "Metoprolol", "Omeprazole", "Sertraline", "Simvastatin"
        ]

        # Tracking variables
        self.iteration = 0
        self.permutation_index = 0
        self.convergence_path: List[float] = []
        self.biological_state_history: List[str] = []
        self.gene_expression = self.initialize_fractal_genes()
        self.disease_risk = self.initialize_fractal_diseases()

        # Visualization data
        self.visualization_data: Dict[str, List[Any]] = {
            'gene_expression': [], 'disease_risk': [],
            'biological_state': [], 'fractal_dimensions': [],
            'drug_efficacy': [], 'drug_safety': [], 'drug_side_effects': []
        }

    # ---------------------------- Generators / init ----------------------------
    def initialize_fractal_dots(self, n_dots: int, n_features: int) -> np.ndarray:
        """Create an (n_dots x n_features) array with small fractal perturbations."""
        base_pattern = self.generate_fractal_pattern(n_features, self.fractal_level)
        dots = np.zeros((n_dots, n_features), dtype=np.float32)

        for i in range(n_dots):
            fractal_noise = self.generate_fractal_noise(n_features, self.fractal_level)
            dot_01 = np.clip(base_pattern + fractal_noise * 0.1, 0.001, 0.999)
            dots[i, :] = dot_01

        return dots

    def initialize_fractal_regions(self, n_regions: int) -> List[np.ndarray]:
        regions = []
        for _ in range(n_regions):
            centroid = np.array([
                self.fractal_value(0.2, 0.8),
                self.fractal_value(0.2, 0.8),
                self.fractal_value(0.2, 0.8),
                self.fractal_value(0.2, 0.8),
                self.fractal_value(0.0001, 0.1)
            ], dtype=float)
            regions.append(centroid)
        return regions

    def initialize_fractal_genes(self) -> Dict[str, float]:
        return {gene: float(self.fractal_value(0.3, 0.7)) for gene in GENES}

    def initialize_fractal_diseases(self) -> Dict[str, float]:
        return {disease: float(self.fractal_value(0.05, 0.3)) for disease in DISEASES}

    # ---------------------------- Fractal utilities ----------------------------
    def generate_fractal_pattern(self, length: int, levels: int) -> np.ndarray:
        """Midpoint displacement-like generator but robust for small lengths.
        Returns values in [0,1].
        """
        if length <= 0:
            return np.array([])
        if length == 1:
            return np.array([self.random_state.rand()])

        pattern = np.zeros(length, dtype=float)
        pattern[0] = self.random_state.rand()
        pattern[-1] = self.random_state.rand()

        scale = 1.0
        step = length - 1

        # iterative midpoint refinement
        while step > 1 and levels > 0:
            half = step // 2
            for i in range(half, length, step):
                left = i - half
                right = i + half
                if right >= length:
                    right = length - 1
                midpoint = 0.5 * (pattern[left] + pattern[right])
                perturb = self.random_state.uniform(-scale, scale)
                pattern[i] = midpoint + perturb
            scale *= self.fractal_scaling
            step = half
            levels -= 1

        # Linear interpolation for any remaining zeros
        zeros = np.where(pattern == 0)[0]
        if zeros.size:
            for idx in zeros:
                # interpolate between nearest non-zero neighbors
                left_idx = idx - 1
                while left_idx >= 0 and pattern[left_idx] == 0:
                    left_idx -= 1
                right_idx = idx + 1
                while right_idx < length and pattern[right_idx] == 0:
                    right_idx += 1
                left_val = pattern[left_idx] if left_idx >= 0 else self.random_state.rand()
                right_val = pattern[right_idx] if right_idx < length else self.random_state.rand()
                t = 0.5
                pattern[idx] = (1 - t) * left_val + t * right_val

        # Normalize
        mn, mx = np.min(pattern), np.max(pattern)
        if mx > mn:
            pattern = (pattern - mn) / (mx - mn)
        else:
            pattern = np.clip(pattern, 0, 1)

        return pattern

    def generate_fractal_noise(self, length: int, levels: int) -> np.ndarray:
        """Generate fractal noise with values in [-1,1]."""
        pattern = self.generate_fractal_pattern(length, levels)
        return 2 * pattern - 1

    def fractal_value(self, low: float, high: float) -> float:
        """Generate a fractal value in [low, high]."""
        return low + (high - low) * self.fractal_value_01()

    def fractal_value_01(self) -> float:
        """Generate a fractal value in [0,1]."""
        return self.generate_fractal_pattern(1, self.fractal_level)[0]

    # ---------------------------- Core 5D dynamics ----------------------------
    def update_5d_coordinates(self) -> None:
        """Update the 5D coordinates based on current state and history."""
        # Base dynamics with fractal perturbations
        self.At = np.clip(self.At + self.fractal_value(-0.05, 0.05), 0.1, 1.0)
        self.Ab = np.clip(self.Ab + self.fractal_value(-0.05, 0.05), 0.1, 1.0)
        self.Ex = np.clip(self.Ex + self.fractal_value(-0.05, 0.05), 0.1, 1.0)
        self.T = np.clip(self.T + self.fractal_value(-0.05, 0.05), 0.1, 1.0)

        # Cn is special - it represents consciousness and requires more complex dynamics
        cn_perturbation = self.fractal_value(-0.1, 0.1)

        # Cn increases with system complexity and decreases with randomness
        complexity = np.std([self.At, self.Ab, self.Ex, self.T])
        randomness = abs(cn_perturbation)

        if complexity > 0.2:
            cn_perturbation += 0.05
        if randomness > 0.15:
            cn_perturbation -= 0.03

        self.Cn = np.clip(self.Cn + cn_perturbation, 0.001, 1.0)

        # Valence is influenced by all coordinates
        self.valence = np.clip(
            0.2 * self.At + 0.2 * self.Ab + 0.2 * self.Ex + 0.2 * self.T + 0.2 * self.Cn,
            0.1, 0.9
        )

        # Track convergence
        self.convergence_path.append(self.Cn)

    def update_biological_state(self) -> None:
        """Update the biological state based on 5D coordinates."""
        # Calculate state based on coordinate balance
        balance = abs(self.At - self.Ab) + abs(self.Ex - self.T)

        if balance < 0.2 and self.Cn > 0.6:
            self.biological_state = "HARMONY"
        elif balance > 0.5 and self.Cn < 0.4:
            self.biological_state = "CHAOS"
        elif self.At > 0.8 and self.Ab < 0.3:
            self.biological_state = "GROWTH"
        elif self.Ab > 0.8 and self.At < 0.3:
            self.biological_state = "DEFENSE"
        elif self.Ex > 0.8 and self.T < 0.3:
            self.biological_state = "ENERGY"
        elif self.T > 0.8 and self.Ex < 0.3:
            self.biological_state = "MAINTENANCE"
        else:
            self.biological_state = "BALANCE"

        self.biological_state_history.append(self.biological_state)

    def update_gene_expression(self) -> None:
        """Update gene expression based on biological state and 5D coordinates."""
        for gene in self.gene_expression:
            # Base expression influenced by biological state
            if self.biological_state == "HARMONY":
                change = self.fractal_value(-0.02, 0.05)
            elif self.biological_state == "CHAOS":
                change = self.fractal_value(-0.05, 0.02)
            elif self.biological_state == "GROWTH":
                change = self.fractal_value(0.01, 0.08) if gene in ["TP53", "BRCA1", "VEGF"] else self.fractal_value(-0.03, 0.03)
            elif self.biological_state == "DEFENSE":
                change = self.fractal_value(0.01, 0.08) if gene in ["HLA-DRB1", "CFTR"] else self.fractal_value(-0.03, 0.03)
            elif self.biological_state == "ENERGY":
                change = self.fractal_value(0.01, 0.08) if gene in ["APOE", "CYP2D6"] else self.fractal_value(-0.03, 0.03)
            elif self.biological_state == "MAINTENANCE":
                change = self.fractal_value(0.01, 0.08) if gene in ["EGFR", "OPRM1"] else self.fractal_value(-0.03, 0.03)
            else:  # BALANCE
                change = self.fractal_value(-0.03, 0.03)

            # Additional influence from 5D coordinates
            coordinate_influence = 0.1 * (self.At - 0.5) + 0.1 * (self.Ab - 0.5) + 0.1 * (self.Ex - 0.5) + 0.1 * (self.T - 0.5)

            self.gene_expression[gene] = np.clip(
                self.gene_expression[gene] + change + coordinate_influence,
                0.0, 1.0
            )

        # Store for visualization
        self.visualization_data['gene_expression'].append(list(self.gene_expression.values()))

    def update_disease_risk(self) -> None:
        """Update disease risk based on gene expression and biological state."""
        for disease in self.disease_risk:
            # Base risk influenced by biological state
            if self.biological_state == "HARMONY":
                change = self.fractal_value(-0.02, 0.01)
            elif self.biological_state == "CHAOS":
                change = self.fractal_value(0.01, 0.05)
            elif self.biological_state == "GROWTH":
                change = self.fractal_value(0.01, 0.04) if disease in ["Cancer", "Obesity"] else self.fractal_value(-0.02, 0.02)
            elif self.biological_state == "DEFENSE":
                change = self.fractal_value(0.01, 0.04) if disease in ["HIV/AIDS", "COVID-19"] else self.fractal_value(-0.02, 0.02)
            elif self.biological_state == "ENERGY":
                change = self.fractal_value(0.01, 0.04) if disease in ["Diabetes", "Obesity"] else self.fractal_value(-0.02, 0.02)
            elif self.biological_state == "MAINTENANCE":
                change = self.fractal_value(0.01, 0.04) if disease in ["Arthritis", "Osteoporosis"] else self.fractal_value(-0.02, 0.02)
            else:  # BALANCE
                change = self.fractal_value(-0.02, 0.02)

            # Additional influence from gene expression
            gene_influence = 0.0
            if disease == "Cancer":
                gene_influence = 0.1 * (self.gene_expression["TP53"] - 0.5) + 0.1 * (self.gene_expression["BRCA1"] - 0.5)
            elif disease == "Alzheimer":
                gene_influence = 0.2 * (self.gene_expression["APOE"] - 0.5)
            elif disease == "Diabetes":
                gene_influence = 0.1 * (self.gene_expression["CYP2D6"] - 0.5)

            self.disease_risk[disease] = np.clip(
                self.disease_risk[disease] + change + gene_influence,
                0.0, 1.0
            )

        # Store for visualization
        self.visualization_data['disease_risk'].append(list(self.disease_risk.values()))

    # ---------------------------- Drug discovery methods ----------------------------
    def create_disease_profile(self) -> Dict[str, float]:
        """Create a disease profile based on current biological state"""
        # Focus on diseases with highest risk
        high_risk_diseases = sorted(
            self.disease_risk.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Create a profile with weights based on risk level
        profile = {}
        for disease, risk in high_risk_diseases:
            # Map disease to relevant drug targets
            if disease in ["Cancer", "HIV/AIDS", "COVID-19"]:
                targets = ["Kinases", "Proteases", "Epigenetic Enzymes"]
            elif disease in ["Alzheimer", "Parkinson", "Multiple Sclerosis"]:
                targets = ["GPCRs", "Ion Channels", "Nuclear Receptors"]
            elif disease in ["Diabetes", "Obesity", "Heart Disease"]:
                targets = ["GPCRs", "Transporters", "Nuclear Receptors"]
            elif disease in ["Depression", "Autism", "Epilepsy"]:
                targets = ["GPCRs", "Ion Channels", "Transporters"]
            else:
                targets = ["GPCRs", "Kinases"]

            # Assign weights based on risk
            for target in targets:
                if target not in profile:
                    profile[target] = 0.0
                profile[target] += risk / len(targets)

        # Normalize
        total = sum(profile.values())
        if total > 0:
            profile = {k: v/total for k, v in profile.items()}

        self.current_disease_profile = profile
        return profile

    def discover_drug_candidates(self, n_candidates: int = 10) -> List[Dict[str, Any]]:
        """Discover potential drug candidates for the current disease profile"""
        disease_profile = self.create_disease_profile()

        # Fetch external data about known drugs for this disease profile
        disease_terms = list(disease_profile.keys())
        external_data = []
        for term in disease_terms:
            data = self.data_fetcher.fetch_ncbi_data(f"{term} drug treatment", db="pubmed", retmax=5)
            external_data.extend(data)

        # Perform virtual screening
        candidates = self.drug_screener.high_throughput_screen(
            n_candidates,
            disease_profile,
            efficacy_threshold=0.5,
            safety_threshold=0.8,
            known_drugs=self.known_drugs
        )

        self.drug_candidates = candidates

        # Store for visualization
        if candidates:
            self.visualization_data['drug_efficacy'].append([c['efficacy_score'] for c in candidates])
            self.visualization_data['drug_safety'].append([c['safety_margin'] for c in candidates])
            self.visualization_data['drug_side_effects'].append([sum(c['side_effects'].values()) for c in candidates])

        return candidates

    def optimize_dosage(self, candidate_idx: int = 0) -> Dict[str, Any]:
        """Optimize dosage for a drug candidate"""
        if not self.drug_candidates or candidate_idx >= len(self.drug_candidates):
            return {}

        candidate = self.drug_candidates[candidate_idx]
        compound = candidate['compound_properties']

        # Test different doses
        doses = [1, 5, 10, 20, 50, 100]  # mg
        results = []

        for dose in doses:
            pkpd_results = self.drug_screener.pkpd_model.simulate_dose_response(
                dose,
                compound['molecular_properties'],
                compound['target_affinity'],
                compound['selectivity'],
                compound['mechanism'],
                self.current_disease_profile
            )

            # Calculate therapeutic window
            ld50 = candidate['ld50']
            therapeutic_index = ld50 / dose if dose > 0 else 0

            results.append({
                'dose': dose,
                'auc': pkpd_results['auc'],
                'cmax': pkpd_results['cmax'],
                'tmax': pkpd_results['tmax'],
                'half_life': pkpd_results['half_life'],
                'therapeutic_index': therapeutic_index
            })

        # Find optimal dose (max therapeutic index with reasonable exposure)
        optimal_idx = 0
        best_score = 0

        for i, res in enumerate(results):
            score = res['therapeutic_index'] * min(1.0, res['auc'] / 100)
            if score > best_score:
                best_score = score
                optimal_idx = i

        self.optimal_dose = results[optimal_idx]
        return results[optimal_idx]

    # ---------------------------- Main simulation loop ----------------------------
    def simulate(self, n_iterations: int = 100) -> None:
        """Run the main simulation loop."""
        for i in range(n_iterations):
            self.iteration = i

            # Update core 5D dynamics
            self.update_5d_coordinates()
            self.update_biological_state()

            # Update biological systems
            self.update_gene_expression()
            self.update_disease_risk()

            # Periodically discover drug candidates
            if i % 20 == 0:
                self.discover_drug_candidates(15)

            # Periodically optimize dosage
            if i % 25 == 0 and self.drug_candidates:
                self.optimize_dosage(0)

            # Store fractal dimensions for visualization
            self.visualization_data['fractal_dimensions'].append([self.At, self.Ab, self.Ex, self.T, self.Cn])
            self.visualization_data['biological_state'].append(self.biological_state)

            # Log progress
            if i % 10 == 0:
                logger.info(
                    f"Iteration {i}: State={self.biological_state}, "
                    f"Coordinates=[{self.At:.2f}, {self.Ab:.2f}, {self.Ex:.2f}, {self.T:.2f}, {self.Cn:.2f}], "
                    f"Valence={self.valence:.2f}"
                )

                if self.drug_candidates:
                    logger.info(
                        f"Top drug candidate: Efficacy={self.drug_candidates[0]['efficacy_score']:.2f}, "
                        f"Safety={self.drug_candidates[0]['safety_margin']:.2f}"
                    )

    # ---------------------------- Visualization methods ----------------------------
    def visualize_simulation(self) -> None:
        """Create comprehensive visualization of the simulation results."""
        try:
            fig = plt.figure(figsize=(20, 16))
            gs = gridspec.GridSpec(4, 3, figure=fig)

            # 1. 5D Coordinates over time
            ax1 = fig.add_subplot(gs[0, 0])
            coords_data = np.array(self.visualization_data['fractal_dimensions'])
            iterations = range(len(coords_data))
            ax1.plot(iterations, coords_data[:, 0], label='At')
            ax1.plot(iterations, coords_data[:, 1], label='Ab')
            ax1.plot(iterations, coords_data[:, 2], label='Ex')
            ax1.plot(iterations, coords_data[:, 3], label='T')
            ax1.plot(iterations, coords_data[:, 4], label='Cn')
            ax1.set_title('5D Coordinates Over Time')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Biological State
            ax2 = fig.add_subplot(gs[0, 1])
            states = self.visualization_data['biological_state']
            state_codes = [hash(state) % 10 for state in states]  # Simple coding for visualization
            ax2.plot(iterations, state_codes, 'o-', markersize=3)
            ax2.set_title('Biological State Over Time')
            ax2.set_xlabel('Iteration')
            ax2.set_yticks(range(10))
            ax2.set_yticklabels(list(set(states))[:10] + [''] * (10 - min(10, len(set(states)))))
            ax2.grid(True, alpha=0.3)

            # 3. Gene Expression Heatmap
            ax3 = fig.add_subplot(gs[0, 2])
            gene_data = np.array(self.visualization_data['gene_expression']).T
            im = ax3.imshow(gene_data, aspect='auto', cmap='viridis')
            ax3.set_title('Gene Expression Over Time')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Gene')
            ax3.set_yticks(range(len(GENES)))
            ax3.set_yticklabels(GENES, fontsize=8)
            plt.colorbar(im, ax=ax3)

            # 4. Disease Risk Heatmap
            ax4 = fig.add_subplot(gs[1, 0])
            disease_data = np.array(self.visualization_data['disease_risk']).T
            im = ax4.imshow(disease_data, aspect='auto', cmap='plasma')
            ax4.set_title('Disease Risk Over Time')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Disease')
            ax4.set_yticks(range(len(DISEASES)))
            ax4.set_yticklabels(DISEASES, fontsize=6)
            plt.colorbar(im, ax=ax4)

            # 5. Drug Discovery Results
            if self.visualization_data['drug_efficacy']:
                ax5 = fig.add_subplot(gs[1, 1])
                efficacy_data = self.visualization_data['drug_efficacy']
                # Plot efficacy of top candidate over screening rounds
                top_candidate_eff = [eff[0] for eff in efficacy_data if len(eff) > 0]
                screening_rounds = range(len(top_candidate_eff))
                ax5.plot(screening_rounds, top_candidate_eff, 'o-', label='Top Candidate')
                ax5.set_title('Drug Efficacy Over Screening Rounds')
                ax5.set_xlabel('Screening Round')
                ax5.set_ylabel('Efficacy Score')
                ax5.legend()
                ax5.grid(True, alpha=0.3)

                ax6 = fig.add_subplot(gs[1, 2])
                safety_data = self.visualization_data['drug_safety']
                top_candidate_safe = [safe[0] for safe in safety_data if len(safe) > 0]
                screening_rounds = range(len(top_candidate_safe))
                ax6.plot(screening_rounds, top_candidate_safe, 'o-', label='Top Candidate')
                ax6.set_title('Drug Safety Over Screening Rounds')
                ax6.set_xlabel('Screening Round')
                ax6.set_ylabel('Safety Margin')
                ax6.legend()
                ax6.grid(True, alpha=0.3)

            # 6. 3D projection of 5D space (At, Ab, Cn)
            ax7 = fig.add_subplot(gs[2, 0], projection='3d')
            ax7.scatter(coords_data[:, 0], coords_data[:, 1], coords_data[:, 4],
                       c=range(len(coords_data)), cmap='viridis', alpha=0.6)
            ax7.set_xlabel('At')
            ax7.set_ylabel('Ab')
            ax7.set_zlabel('Cn')
            ax7.set_title('3D Projection of 5D Space')

            # 7. Convergence of Cn
            ax8 = fig.add_subplot(gs[2, 1])
            ax8.plot(iterations, coords_data[:, 4], label='Cn')
            ax8.set_title('Consciousness (Cn) Convergence')
            ax8.set_xlabel('Iteration')
            ax8.set_ylabel('Cn Value')
            ax8.legend()
            ax8.grid(True, alpha=0.3)

            # 8. Drug Properties Radar Chart (if we have candidates)
            if self.drug_candidates:
                ax9 = fig.add_subplot(gs[2, 2], polar=True)
                candidate = self.drug_candidates[0]
                compound = candidate['compound_properties']

                # Properties to display
                categories = ['Efficacy', 'Safety', 'Solubility', 'Permeability', 'Stability', 'Selectivity']
                values = [
                    candidate['efficacy_score'],
                    candidate['safety_margin'] / 10,  # Scale to similar range
                    compound['solubility'] / 10,
                    (compound['permeability'] + 5) / 7,  # Scale from -5 to 2 to 0-1
                    compound['stability']['half_life_hours'] / 1000,
                    compound['selectivity']
                ]

                N = len(categories)
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the circle
                values += values[:1]

                ax9.plot(angles, values, 'o-', linewidth=2)
                ax9.fill(angles, values, alpha=0.25)
                ax9.set_thetagrids([a * 180/np.pi for a in angles[:-1]], categories)
                ax9.set_title('Top Drug Candidate Properties')
                ax9.grid(True)

            # 9. Side Effects Profile
            if self.drug_candidates and 'side_effects' in self.drug_candidates[0]:
                ax10 = fig.add_subplot(gs[3, :])
                side_effects = self.drug_candidates[0]['side_effects']
                effects = list(side_effects.keys())
                probabilities = list(side_effects.values())

                y_pos = np.arange(len(effects))
                ax10.barh(y_pos, probabilities, align='center', alpha=0.7)
                ax10.set_yticks(y_pos)
                ax10.set_yticklabels(effects)
                ax10.set_xlabel('Probability')
                ax10.set_title('Predicted Side Effects Profile')
                ax10.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            plt.savefig('consciousleaf5d_simulation.png', dpi=150, bbox_inches='tight')
            plt.show()

        except Exception as e:
            logger.error(f"Visualization error: {e}")
            # Fallback simple plot
            plt.figure(figsize=(12, 8))
            coords_data = np.array(self.visualization_data['fractal_dimensions'])
            iterations = range(len(coords_data))
            plt.plot(iterations, coords_data[:, 0], label='At')
            plt.plot(iterations, coords_data[:, 1], label='Ab')
            plt.plot(iterations, coords_data[:, 2], label='Ex')
            plt.plot(iterations, coords_data[:, 3], label='T')
            plt.plot(iterations, coords_data[:, 4], label='Cn')
            plt.title('5D Coordinates Over Time')
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('consciousleaf5d_fallback.png', dpi=150, bbox_inches='tight')
            plt.show()


# ---------------------------- Main execution ----------------------------
if __name__ == "__main__":
    # Initialize and run simulation
    simulator = ConsciousLeaf5D(seed=42)

    logger.info("Starting ConsciousLeaf5D Drug Discovery Simulation")
    logger.info("=" * 60)

    # Run simulation
    simulator.simulate(n_iterations=100)

    # Display results
    logger.info("Simulation completed!")
    logger.info("=" * 60)

    if simulator.drug_candidates:
        top_candidate = simulator.drug_candidates[0]
        logger.info(f"Top Drug Candidate:")
        logger.info(f"  Efficacy Score: {top_candidate['efficacy_score']:.3f}")
        logger.info(f"  Safety Margin: {top_candidate['safety_margin']:.3f}")
        logger.info(f"  Therapeutic Index: {top_candidate['therapeutic_index']:.3f}")
        logger.info(f"  LD50: {top_candidate['ld50']:.1f} mg/kg")
        logger.info(f"  ED50: {top_candidate['ed50']:.1f} mg")

        if simulator.optimal_dose:
            logger.info(f"Optimal Dose: {simulator.optimal_dose['dose']} mg")
            logger.info(f"  AUC: {simulator.optimal_dose['auc']:.1f} mg·h/L")
            logger.info(f"  Cmax: {simulator.optimal_dose['cmax']:.2f} mg/L")
            logger.info(f"  Tmax: {simulator.optimal_dose['tmax']:.1f} h")
            logger.info(f"  Half-life: {simulator.optimal_dose['half_life']:.1f} h")

    # Generate visualization
    logger.info("Generating visualization...")
    simulator.visualize_simulation()

    logger.info("Done!")

    OUTPUT:

    2025-09-09 05:22:17,662 - INFO - Starting ConsciousLeaf5D Drug Discovery Simulation
2025-09-09 05:22:17,662 - INFO - Starting ConsciousLeaf5D Drug Discovery Simulation
INFO:ConsciousLeaf5D:Starting ConsciousLeaf5D Drug Discovery Simulation
2025-09-09 05:22:17,665 - INFO - ============================================================
2025-09-09 05:22:17,665 - INFO - ============================================================
INFO:ConsciousLeaf5D:============================================================
/tmp/ipython-input-3013127929.py:256: DeprecationWarning: `trapz` is deprecated. Use `trapezoid` instead, or one of the numerical integration functions in `scipy.integrate`.
  auc = np.trapz(concentration, time_points)
2025-09-09 05:22:20,742 - INFO - Iteration 0: State=BALANCE, Coordinates=[0.74, 0.70, 0.72, 0.66, 0.33], Valence=0.63
2025-09-09 05:22:20,742 - INFO - Iteration 0: State=BALANCE, Coordinates=[0.74, 0.70, 0.72, 0.66, 0.33], Valence=0.63
INFO:ConsciousLeaf5D:Iteration 0: State=BALANCE, Coordinates=[0.74, 0.70, 0.72, 0.66, 0.33], Valence=0.63
2025-09-09 05:22:20,752 - INFO - Iteration 10: State=BALANCE, Coordinates=[0.84, 0.62, 0.77, 0.66, 0.50], Valence=0.68
2025-09-09 05:22:20,752 - INFO - Iteration 10: State=BALANCE, Coordinates=[0.84, 0.62, 0.77, 0.66, 0.50], Valence=0.68
INFO:ConsciousLeaf5D:Iteration 10: State=BALANCE, Coordinates=[0.84, 0.62, 0.77, 0.66, 0.50], Valence=0.68
2025-09-09 05:22:21,589 - INFO - Iteration 20: State=BALANCE, Coordinates=[0.89, 0.60, 0.72, 0.74, 0.86], Valence=0.76
2025-09-09 05:22:21,589 - INFO - Iteration 20: State=BALANCE, Coordinates=[0.89, 0.60, 0.72, 0.74, 0.86], Valence=0.76
INFO:ConsciousLeaf5D:Iteration 20: State=BALANCE, Coordinates=[0.89, 0.60, 0.72, 0.74, 0.86], Valence=0.76
2025-09-09 05:22:21,597 - INFO - Iteration 30: State=BALANCE, Coordinates=[1.00, 0.51, 0.60, 0.64, 0.92], Valence=0.73
2025-09-09 05:22:21,597 - INFO - Iteration 30: State=BALANCE, Coordinates=[1.00, 0.51, 0.60, 0.64, 0.92], Valence=0.73
INFO:ConsciousLeaf5D:Iteration 30: State=BALANCE, Coordinates=[1.00, 0.51, 0.60, 0.64, 0.92], Valence=0.73
2025-09-09 05:22:21,672 - INFO - Iteration 40: State=BALANCE, Coordinates=[0.94, 0.38, 0.65, 0.66, 0.91], Valence=0.71
2025-09-09 05:22:21,672 - INFO - Iteration 40: State=BALANCE, Coordinates=[0.94, 0.38, 0.65, 0.66, 0.91], Valence=0.71
INFO:ConsciousLeaf5D:Iteration 40: State=BALANCE, Coordinates=[0.94, 0.38, 0.65, 0.66, 0.91], Valence=0.71
2025-09-09 05:22:21,681 - INFO - Iteration 50: State=BALANCE, Coordinates=[0.83, 0.45, 0.76, 0.66, 0.99], Valence=0.74
2025-09-09 05:22:21,681 - INFO - Iteration 50: State=BALANCE, Coordinates=[0.83, 0.45, 0.76, 0.66, 0.99], Valence=0.74
INFO:ConsciousLeaf5D:Iteration 50: State=BALANCE, Coordinates=[0.83, 0.45, 0.76, 0.66, 0.99], Valence=0.74
2025-09-09 05:22:21,760 - INFO - Iteration 60: State=BALANCE, Coordinates=[0.78, 0.37, 0.63, 0.67, 0.48], Valence=0.59
2025-09-09 05:22:21,760 - INFO - Iteration 60: State=BALANCE, Coordinates=[0.78, 0.37, 0.63, 0.67, 0.48], Valence=0.59
INFO:ConsciousLeaf5D:Iteration 60: State=BALANCE, Coordinates=[0.78, 0.37, 0.63, 0.67, 0.48], Valence=0.59
2025-09-09 05:22:21,769 - INFO - Iteration 70: State=BALANCE, Coordinates=[0.69, 0.26, 0.66, 0.64, 0.45], Valence=0.54
2025-09-09 05:22:21,769 - INFO - Iteration 70: State=BALANCE, Coordinates=[0.69, 0.26, 0.66, 0.64, 0.45], Valence=0.54
INFO:ConsciousLeaf5D:Iteration 70: State=BALANCE, Coordinates=[0.69, 0.26, 0.66, 0.64, 0.45], Valence=0.54
2025-09-09 05:22:22,175 - INFO - Iteration 80: State=BALANCE, Coordinates=[0.62, 0.20, 0.92, 0.59, 0.97], Valence=0.66
2025-09-09 05:22:22,175 - INFO - Iteration 80: State=BALANCE, Coordinates=[0.62, 0.20, 0.92, 0.59, 0.97], Valence=0.66
INFO:ConsciousLeaf5D:Iteration 80: State=BALANCE, Coordinates=[0.62, 0.20, 0.92, 0.59, 0.97], Valence=0.66
2025-09-09 05:22:22,185 - INFO - Iteration 90: State=BALANCE, Coordinates=[0.70, 0.17, 0.90, 0.57, 1.00], Valence=0.67
2025-09-09 05:22:22,185 - INFO - Iteration 90: State=BALANCE, Coordinates=[0.70, 0.17, 0.90, 0.57, 1.00], Valence=0.67
INFO:ConsciousLeaf5D:Iteration 90: State=BALANCE, Coordinates=[0.70, 0.17, 0.90, 0.57, 1.00], Valence=0.67
2025-09-09 05:22:22,194 - INFO - Simulation completed!
2025-09-09 05:22:22,194 - INFO - Simulation completed!
INFO:ConsciousLeaf5D:Simulation completed!
2025-09-09 05:22:22,197 - INFO - ============================================================
2025-09-09 05:22:22,197 - INFO - ============================================================
INFO:ConsciousLeaf5D:============================================================
2025-09-09 05:22:22,199 - INFO - Generating visualization...
2025-09-09 05:22:22,199 - INFO - Generating visualization...
INFO:ConsciousLeaf5D:Generating visualization...

<img width="1954" height="1226" alt="image" src="https://github.com/user-attachments/assets/9e05b1b7-088e-4fca-8904-dc9e78662f32" />

2025-09-09 05:22:24,893 - INFO - Done!
2025-09-09 05:22:24,893 - INFO - Done!
INFO:ConsciousLeaf5D:Done!
