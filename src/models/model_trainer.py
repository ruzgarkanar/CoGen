import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from typing import List, Dict
import logging
import re

class QADataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer):
        self.tokenizer = tokenizer
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['question'],
            item['context'],
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten().long(),  
            'attention_mask': encoding['attention_mask'].flatten().long(),  
            'start_positions': torch.tensor(item['start_position'], dtype=torch.long),
            'end_positions': torch.tensor(item['end_position'], dtype=torch.long)
        }

class ModelTrainer:
    def __init__(self, model_manager):
        self.logger = logging.getLogger(__name__)
        self.model_manager = model_manager
        
        self.device = torch.device('cpu')
        self.logger.info(f"Using device: {self.device} for training")
        
        self.model_manager.model = self.model_manager.model.to(self.device)

    def prepare_training_data(self, documents: List[Dict]) -> List[Dict]:
        self.logger.info(f"Processing {len(documents)} documents for training data")
        qa_pairs = []

        for doc in documents:
            try:
                content = doc['content']
                metadata = doc['metadata']
                
                if 'entities' in metadata:
                    for entity_type, entities in metadata['entities'].items():
                        for entity in entities:
                            if isinstance(entity, dict) and 'text' in entity and 'context' in entity:
                                qa_pair = {
                                    'question': self._generate_entity_question(entity_type, entity['text']),
                                    'context': entity['context'],
                                    'answer': entity['text'],
                                    'start_position': entity['context'].find(entity['text']),
                                    'end_position': entity['context'].find(entity['text']) + len(entity['text'])
                                }
                                if qa_pair['start_position'] != -1:
                                    qa_pairs.append(qa_pair)

                if 'technical_info' in metadata:
                    for tech_type, terms in metadata['technical_info'].items():
                        if isinstance(terms, list):
                            for term in terms:
                                qa_pairs.append({
                                    'question': f"What is the {tech_type} related to {term}?",
                                    'context': content,
                                    'answer': term,
                                    'start_position': content.find(term),
                                    'end_position': content.find(term) + len(term)
                                })

                sections = self._split_into_sections(content)
                for section in sections:
                    if section['title'] and section['content']:
                        qa_pairs.append({
                            'question': f"What information is provided about {section['title']}?",
                            'context': section['content'],
                            'answer': section['content'].split('.')[0], 
                            'start_position': 0,
                            'end_position': len(section['content'].split('.')[0])
                        })

                self.logger.info(f"Generated {len(qa_pairs)} QA pairs for document")
                
            except Exception as e:
                self.logger.error(f"Error processing document: {str(e)}")
                continue

        valid_pairs = [
            pair for pair in qa_pairs 
            if pair['question'] and pair['answer'] and 
            pair['start_position'] != -1 and
            len(pair['context']) > 0
        ]

        self.logger.info(f"Generated total of {len(valid_pairs)} valid QA pairs")
        
        if valid_pairs:
            self.logger.debug("Sample QA pairs:")
            for pair in valid_pairs[:3]:
                self.logger.debug(f"Q: {pair['question']}")
                self.logger.debug(f"A: {pair['answer']}")
                self.logger.debug(f"Context: {pair['context'][:100]}...")
                self.logger.debug("---")

        return valid_pairs

    def _split_into_sections(self, text: str) -> List[Dict]:
        sections = []
        lines = text.split('\n')
        current_section = {'title': '', 'content': ''}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if any([
                line.isupper(),
                line.startswith('#'),
                re.match(r'^[\d\.]+\s+[A-Z]', line),
                len(line.split()) <= 5 and line.istitle()
            ]):
                if current_section['content']:
                    sections.append(current_section)
                current_section = {'title': line, 'content': ''}
            else:
                current_section['content'] += line + ' '
        
        if current_section['content']:
            sections.append(current_section)
            
        return sections

    def _generate_entity_question(self, entity_type: str, entity_text: str) -> str:
        questions = {
            'ORG': [
                f"What is {entity_text}?",
                f"Can you describe {entity_text}?",
                f"What does {entity_text} do?"
            ],
            'PERSON': [
                f"Who is {entity_text}?",
                f"What role does {entity_text} play?",
                f"What is known about {entity_text}?"
            ],
            'CARDINAL': [
                f"What does the number {entity_text} represent?",
                f"What is the significance of {entity_text}?",
                f"What measurement shows {entity_text}?"
            ]
        }
        
        default_questions = [f"What is {entity_text}?"]
        possible_questions = questions.get(entity_type, default_questions)
        return possible_questions[0]

    def train_model(self, training_data: List[Dict], 
                   epochs: int = 50, 
                   batch_size: int = 1,
                   learning_rate: float = 5e-6):  
        try:
            if not training_data:
                raise ValueError("No valid training data provided")
                
            torch.set_num_threads(8) 
            
            dataset = QADataset(training_data, self.model_manager.tokenizer)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False 
            )
            
            self.model_manager.model = self.model_manager.model.cpu()
            
            optimizer = AdamW(
                self.model_manager.model.parameters(),
                lr=learning_rate,
                eps=1e-8,
                weight_decay=0.001,  
                correct_bias=True
            )
            
            num_training_steps = len(dataloader) * epochs
            num_warmup_steps = num_training_steps // 10
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                epochs=epochs,
                steps_per_epoch=len(dataloader),
                pct_start=0.1,
                anneal_strategy='cos'
            )
            
            max_grad_norm = 0.1

            best_loss = float('inf')
            no_improvement = 0
            
            for epoch in range(epochs):
                total_loss = 0.0
                valid_batches = 0
                
                self.model_manager.model.train()
                
                for batch_idx, batch in enumerate(dataloader):
                    try:
                        batch = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) 
                               for k, v in batch.items()}

                        for param in self.model_manager.model.parameters():
                            if param.grad is not None:
                                if not torch.isfinite(param.grad).all():
                                    param.grad.zero_()

                        outputs = self.model_manager.model(**batch)
                        loss = outputs.loss
                        
                        if torch.isfinite(loss) and loss.item() > 0:
                            scaled_loss = loss / batch_size
                            
                            scaled_loss.backward()

                            torch.nn.utils.clip_grad_norm_(
                                self.model_manager.model.parameters(),
                                max_norm=max_grad_norm
                            )
                            
                            valid_gradients = True
                            for param in self.model_manager.model.parameters():
                                if param.grad is not None:
                                    if not torch.isfinite(param.grad).all():
                                        valid_gradients = False
                                        break

                            if valid_gradients:
                                optimizer.step()
                                scheduler.step()
                                total_loss += loss.item()
                                valid_batches += 1

                                if batch_idx % 10 == 0:
                                    self.logger.info(
                                        f"Epoch {epoch+1}, Batch {batch_idx}, "
                                        f"Loss: {loss.item():.4f}, "
                                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                                    )
                            else:
                                self.logger.warning(
                                    f"Batch {batch_idx}: Invalid gradients detected, skipping..."
                                )
                        else:
                            self.logger.warning(
                                f"Batch {batch_idx}: Invalid loss value {loss.item()}, skipping..."
                            )
                            
                    except Exception as e:
                        self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                        continue
                
                if valid_batches > 0:
                    avg_loss = total_loss / valid_batches
                    self.logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
                    
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        no_improvement = 0
                        self.model_manager.save_model()
                    else:
                        no_improvement += 1
                        if no_improvement >= 5:
                            self.logger.info("Early stopping triggered")
                            break
                else:
                    self.logger.warning(f"Epoch {epoch+1}: No valid batches!")
                    
            self.model_manager.optimize_model()
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

    def _create_question_for_entity(self, entity_type: str, entity_text: str) -> str:
        if not entity_text:
            return ""
            
        question_templates = {
            'PERSON': [
                f"Who is {entity_text}?",
                f"What is {entity_text}'s role?",
            ],
            'ORG': [
                f"What is {entity_text}?",
                f"Can you describe {entity_text}?",
            ],
            'DATE': [
                f"When did {entity_text} occur?",
                f"What happened on {entity_text}?",
            ],
            'LOCATION': [
                f"Where is {entity_text}?",
                f"What can you tell me about {entity_text}?",
            ],
            'PRODUCT': [
                f"What is {entity_text}?",
                f"Can you describe the {entity_text}?",
            ]
        }
        
        templates = question_templates.get(entity_type, [])
        return templates[0] if templates else ""

    def _create_technical_question(self, info_type: str, info: str) -> str:
        question_templates = {
            'measurements': f"What are the measurements for {info}?",
            'specifications': f"What are the specifications of {info}?",
            'product_codes': f"What is the product code {info} for?",
            'technical_terms': f"Can you explain the technical term {info}?"
        }
        
        return question_templates.get(info_type, "")
