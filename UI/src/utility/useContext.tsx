import { useContext } from 'react';
import { Context, DataContextType } from './Context';

export function useData(): DataContextType {
    return useContext(Context);
}