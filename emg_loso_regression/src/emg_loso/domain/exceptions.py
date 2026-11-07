"""Exceções específicas de domínio para violações de contrato, formato de sinal e vazamento de dados."""
from __future__ import annotations


class DomainError(Exception):
    """Exceção base para todos os erros da camada de domínio."""


class ContractViolationError(DomainError):
    """Lançada quando uma entidade ou configuração viola invariantes de protocolo (RFC-0001)."""


class InvalidSignalShapeError(DomainError):
    """Lançada quando uma matriz de EMG ou cinemática viola as dimensões físicas esperadas."""


class DataLeakageViolationError(DomainError):
    """Lançada se houver contaminação entre sujeitos de treino/teste ou entre repetições (rep0/1 vs rep2)."""
